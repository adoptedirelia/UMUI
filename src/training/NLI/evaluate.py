from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
from config import OmniTrainingConfig, OmniEvalConfig
from typing import List, Dict, Any
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import torch.nn.functional as F
import json
from prompt import *
from evaldataset import *
import warnings
import argparse


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="System prompt modified")


class PromptBuilder:
    @staticmethod

    def build_messages_vl(claim: str, video_path: str,config: OmniTrainingConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_SYSTEM_PROMPT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_PROMPT.format(text=claim),
                    },
                    {
                        "type": "video",
                        "video": video_path,
                        "resized_height": 256,
                        "resized_width": 256,
                        "fps": 0.5,
                    },
                ],
            }
        ]

    def build_messages_audio(claim: str, audio_path: str, config: OmniTrainingConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_SYSTEM_PROMPT_AUDIO,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_PROMPT_AUDIO.format(text=claim),
                    },
                    {
                        "type": "audio",
                        "audio": audio_path,
                    },
                ],
            }
        ]

    def build_messages_text(claim: str, sentence: str, config: OmniTrainingConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_SYSTEM_PROMPT_TEXT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": NLI_PROMPT_TEXT.format(sentence=sentence, claim=claim),
                    }
                ]
            }
        ]



def extract_answer(logits: torch.Tensor,config: OmniEvalConfig,processor: Qwen2_5OmniProcessor) -> str:
    # find <answer> and </answer> and return the text between them

    logits = logits[-1,:]
    CONF_tokens = [config.new_token_prefix.format(idx=idx) for idx in range(config.new_token_num)]

    CONF_logits = logits[processor.tokenizer.convert_tokens_to_ids(CONF_tokens)]

    logprob = F.log_softmax(CONF_logits, dim=-1)
    probabilities = torch.exp(logprob)
    scale_values = torch.linspace(0.0, 1.0, config.new_token_num).to(logits.device)
    pred_scores = (probabilities * scale_values).sum(dim=-1)

    return '<answer>' + str(pred_scores.item()) + '</answer>'

class OmniEvaluator:
    def __init__(self, config: OmniEvalConfig):
        self.config = config
        self.modality = config.modality
        self.batch_size = config.batch_size
        self.process_mm_info = process_mm_info
    def create_model(self):
    
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(self.config.processor_path, torch_dtype="bfloat16",device_map=self.config.device,attn_implementation="flash_attention_2",)

        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.processor_path)

        self.model = PeftModel.from_pretrained(self.model, self.config.lora_path)


    def create_dataset(self):
        if self.config.modality == 'text':
            if self.config.dataset_name == 'peopleprofile':
                self.eval_dataset = PeopleProfileDataset(self.config).extract_data()
            if self.config.dataset_name == 'unli':
                self.eval_dataset = UNLI_Dataset(self.config).extract_data()
        elif self.config.modality == 'audio':
            if self.config.dataset_name == 'clotho':
                self.eval_dataset = ClothoDataset(self.config).extract_data()
            if self.config.dataset_name == 'wikivideo':
                self.eval_dataset = WikiVideoDataset(self.config).extract_data()
        elif self.config.modality == 'video':
            if self.config.dataset_name == 'wikivideo':
                self.eval_dataset = WikiVideoDataset(self.config).extract_data()
            if self.config.dataset_name == 'violin':
                self.eval_dataset = ViolinDataset(self.config).extract_data()
        elif self.config.modality == 'omni':
            if self.config.dataset_name == 'wikivideo':
                temp_dataset = WikiVideoDataset(self.config)
                self.eval_dataset = temp_dataset.extract_data()

        if isinstance(self.eval_dataset, dict):
            self.eval_dataset = [item for sublist in self.eval_dataset.values() for item in sublist ]
        else:
            self.eval_dataset = self.eval_dataset

    def process_data(self, llm_inputs: list[dict[str, any]]) -> list[dict[str, any]]:

        if self.modality == 'text':
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(llm_inputs, add_generation_prompt=True, tokenize=False)

            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
        elif self.modality == 'audio':
            USE_AUDIO_IN_VIDEO = True
            text = self.processor.apply_chat_template(llm_inputs, add_generation_prompt=True, tokenize=False)
            audios, images, videos = self.process_mm_info(llm_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
        elif self.modality == 'video':
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(llm_inputs, add_generation_prompt=True, tokenize=False)
            audios, images, videos = self.process_mm_info(llm_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
        elif self.modality == 'omni':

            try:
                USE_AUDIO_IN_VIDEO = True
                text = self.processor.apply_chat_template(llm_inputs, add_generation_prompt=True, tokenize=False)
                audios, images, videos = self.process_mm_info(llm_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

                inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(self.model.device).to(self.model.dtype)
            except:
                USE_AUDIO_IN_VIDEO = False
                text = self.processor.apply_chat_template(llm_inputs, add_generation_prompt=True, tokenize=False)
                audios, images, videos = self.process_mm_info(llm_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

                inputs = self.processor(text=text, videos=videos, return_tensors="pt", padding=True)
                inputs = inputs.to(self.model.device).to(self.model.dtype)

        outputs = self.model(**inputs)
        logits = outputs.logits.detach() 
        del outputs

        return logits

    def filter_data(self, dict_data: dict[str, any], data: list[dict[str, any]]) -> list[dict[str, any]]:
        result = []
        unfolded_data = [item for sublist in dict_data.values() for item in sublist ]
        for i in data:
            # if i is in dict_data, skip
            flag = False
            for j in unfolded_data:
                if i['path'] == j['path'] and i['claim'] == j['claim']:
                    flag = True
                    break
            if not flag:
                result.append(i)


        return result

    def evaluate(self):
        self.model.eval()
        if os.path.exists(self.config.output_path):
            with open(self.config.output_path, 'r') as f:
                output_results = json.load(f)
        else:
            output_results = {}

        if isinstance(self.eval_dataset, dict):
            unfolded_data = [item for sublist in self.eval_dataset.values() for item in sublist ]
        else:
            unfolded_data = self.eval_dataset


        unfolded_data = self.filter_data(output_results, unfolded_data)

        for i in tqdm(range(0, len(unfolded_data), self.config.batch_size)):

            
            llm_inputs = []

            batch_data = unfolded_data[i:i+self.batch_size]

            events = [item["type"] for item in batch_data]

            for event in events:
                if event not in output_results:
                    output_results[event] = []
            if self.modality == 'text':
                claims = [item["claim"] for item in batch_data]
                sentences = [item["sentence"] for item in batch_data]
                for claim, sentence in zip(claims, sentences):
                    messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
                    llm_inputs.append(messages)

            elif self.modality == 'audio':
                claims = [item["claim"] for item in batch_data]
                paths = [item["path"] for item in batch_data]
                for claim, path in zip(claims, paths):
                    messages = PromptBuilder.build_messages_audio(claim, path, self.config)
                    llm_inputs.append(messages)
            elif self.modality == 'video':
                claims = [item["claim"] for item in batch_data]
                paths = [item["path"] for item in batch_data]
                for claim, path in zip(claims, paths):
                    messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                    llm_inputs.append(messages)
            elif self.modality == 'omni':
                claims = [item["claim"] for item in batch_data]
                paths = [item["path"] for item in batch_data]
                for claim, path in zip(claims, paths):
                    messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                    llm_inputs.append(messages)
            
            with torch.no_grad():
                logits = self.process_data(llm_inputs)


            for batch_data, logit in zip(batch_data, logits):
                with torch.no_grad():
                    answer = extract_answer(logit, self.config, self.processor)
                if self.modality == 'text':
                    output_results[batch_data['type']].append({
                        'type': batch_data['type'],
                        'claim': batch_data['claim'],
                        'label': batch_data['label'],
                        'sentence': batch_data['sentence'],
                        'answer': answer,
                        'modality': self.modality,
                    })
                else:
                    output_results[batch_data['type']].append({
                        'type': batch_data['type'],
                        'claim': batch_data['claim'],
                        'label': batch_data['label'],
                        'path': batch_data['path'],
                        'answer': answer,
                        'modality': self.modality,
                    })
                with open(self.config.output_path, 'w') as f:
                    json.dump(output_results, f, indent=4)
            del logits
            torch.cuda.empty_cache()

        return output_results


if __name__ == "__main__":
    config = OmniEvalConfig()
    evaluator = OmniEvaluator(config)
    evaluator.create_dataset()

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data_begin', type=int, default=0)
    parser.add_argument('--data_end', type=int, default=len(evaluator.eval_dataset))
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    config.device = args.device
    config.data_begin = args.data_begin
    config.data_end = args.data_end
    config.batch_size = args.batch_size
    if config.modality == 'video' or config.modality == 'omni':
        os.makedirs(os.path.join(config.output_eval_path), exist_ok=True)
        config.output_path = os.path.join(config.output_eval_path,  f'{config.dataset_name}_{config.modality}_{config.data_begin}_{config.data_end}.json')
    else:
        os.makedirs(os.path.join(config.output_eval_path), exist_ok=True)
        config.output_path = os.path.join(config.output_eval_path,  f'{config.dataset_name}_{config.modality}_{config.data_begin}_{config.data_end}.json')
    evaluator.create_model()
    results = evaluator.evaluate()
    with open(config.output_path, 'w') as f:
        json.dump(results, f, indent=4)