# from vllm import LLM, SamplingParams
from src.prompt_score import INSTRUCTION, PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_AUDIO, INSTRUCTION_AUDIO, PROMPT_AUDIO
from src.synthetic_data.config import AppConfig
import torch
from vllm import LLM, SamplingParams

import warnings
warnings.filterwarnings("ignore")
class PromptBuilder:
    @staticmethod
    def build_messages_vl(claim: str, video_path: str,config: AppConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION + "\n\n" + PROMPT.format(text=claim),
                    },
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": config.min_pixels,
                        "max_pixels": config.max_pixels,
                        "fps": config.fps,
                    },
                ],
            }
        ]
    def build_messages_audio(claim: str, audio_path: str, config: AppConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_AUDIO,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": INSTRUCTION_AUDIO + "\n\n" + PROMPT_AUDIO.format(text=claim),
                    },
                    {
                        "type": "audio",
                        "audio": audio_path,
                    },
                ],
            }
        ]



class Qwen2_5_OmniHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info

        self.config = config
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.model, cache_dir=config.cache_dir, device_map="auto", torch_dtype="auto",attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)
        # get 0-9
        self.score_token = ['0','1','2','3','4','5','6','7','8','9']


    
    def generate(self, claim: str, video_path: str, max_new_tokens: int) -> str:
        messages = PromptBuilder.build_messages_vl(claim, video_path, self.config)
        with torch.no_grad():
            try:
                USE_AUDIO_IN_VIDEO = True
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(self.model.device).to(self.model.dtype)


                text_ids = self.model(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,do_sample=True,temperature=0.7,top_p=0.95)

                # outputs = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            except:
                USE_AUDIO_IN_VIDEO = False
                text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = self.processor(text=text, audio=None, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = inputs.to(self.model.device).to(self.model.dtype)
                text_ids = self.model(**inputs,do_sample=True,temperature=0.7,top_p=0.95)
                # outputs = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logits = text_ids.logits[0,-1,:]
            # choose logits from corresponding score token
            score_logits = logits[self.tokenizer.convert_tokens_to_ids(self.target_token)]
            score = torch.softmax(score_logits, dim=-1)

        return score
    
    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        outputs = []
        claims = [item['claim'] for item in batch_data]
        paths = [item['path'] for item in batch_data]
        messages = []
        for claim, path in zip(claims, paths):
            if self.config.modality == 'audio':
                messages.append(PromptBuilder.build_messages_audio(claim, path, self.config))
            else:
                messages.append(PromptBuilder.build_messages_vl(claim, path, self.config))
        try:
            USE_AUDIO_IN_VIDEO = True


            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            # Batch Inference
            with torch.no_grad():
                text_ids = self.model(**inputs)
        except:
            USE_AUDIO_IN_VIDEO = False

            with torch.no_grad():
                text_ids = self.model(**inputs)
        logits = text_ids.logits[:,-1,:]

        score_logits = logits[:,self.tokenizer.convert_tokens_to_ids(self.score_token)]
        score = torch.softmax(score_logits, dim=-1).tolist()

        return score
