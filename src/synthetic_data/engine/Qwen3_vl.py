# import os
# os.environ['VLLM_USE_V1'] = '0'
# import warnings
# warnings.filterwarnings("ignore")
from vllm import LLM, SamplingParams
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig


class PromptBuilder:
    @staticmethod
    def build_messages_vl(claim: str, path: str,config: AppConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": src.prompt_binary.SYSTEM_PROMPT if config.binary else src.prompt.SYSTEM_PROMPT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": src.prompt_binary.INSTRUCTION if config.binary else src.prompt.INSTRUCTION + "\n\n" + (src.prompt_binary.PROMPT if config.binary else src.prompt.PROMPT).format(text=claim),
                    },
                    {
                        "type": "video",
                        "video": path,
                        "min_pixels": config.min_pixels,
                        "max_pixels": config.max_pixels,
                        "fps": config.fps,
                    },
                ],
            }
        ]

    def build_messages_text(claim: str, sentence: str, config: AppConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": src.prompt_binary.SYSTEM_PROMPT_TEXT if config.binary else src.prompt.SYSTEM_PROMPT_TEXT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": src.prompt_binary.INSTRUCTION_TEXT if config.binary else src.prompt.INSTRUCTION_TEXT + "\n\n" + (src.prompt_binary.PROMPT_TEXT if config.binary else src.prompt.PROMPT_TEXT).format(sentence=sentence, claim=claim),
                    }
                ]
            }
        ]

class Qwen3_VLHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.model, cache_dir=config.cache_dir, device_map="auto",dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)

    def generate(self, claim: str, path: str, max_new_tokens: int) -> str:
        messages = PromptBuilder.build_messages_vl(claim, path, self.config)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors="pt")

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs[0] if outputs else ""

    def generate_batch(self, claims: list[str], paths: list[str], max_new_tokens: int) -> list[str]:
        outputs = []
        for claim, path in zip(claims, paths):
            outputs.append(self.generate(claim, path, max_new_tokens))
        return outputs

class Qwen3_VLVLLMInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.config = config
        self.llm = LLM(
            model=config.model,
            limit_mm_per_prompt={"video": 1},
            max_model_len=8192,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
            enable_prefix_caching=True,
            disable_custom_all_reduce=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.video_cache = {}

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        if self.config.modality == 'video':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            messages = PromptBuilder.build_messages_vl(claim, path, self.config)
        elif self.config.modality == 'text':
            sentence = [item["sentence"] for item in batch_data]
            claim = [item["claim"] for item in batch_data]
            messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(messages,image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        llm_inputs = {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }
        outputs = self.llm.generate(llm_inputs, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p))
        return outputs[0].outputs[0].text
    
    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        llm_inputs = []
        if self.config.modality == 'video':
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):
                messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = self.process_vision_info(messages,image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)

                mm_data = {}
                if image_inputs is not None:
                    mm_data['image'] = image_inputs
                if video_inputs is not None:
                    mm_data['video'] = video_inputs
                llm_input = {
                    'prompt': text,
                    'multi_modal_data': mm_data,
                    'mm_processor_kwargs': video_kwargs
                }
                llm_inputs.append(llm_input)
        elif self.config.modality == 'text':
            sentences = [item["sentence"] for item in batch_data]
            claims = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentences, claims):
                messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                llm_input = {
                    'prompt': text,
                    'multi_modal_data': {},
                    'mm_processor_kwargs': {}
                }
                llm_inputs.append(llm_input)

        outputs = self.llm.generate(llm_inputs, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p))

        if self.config.response_num > 1:
            res = []
            for output in outputs:
                temp = []
                for output_ in output.outputs:
                    temp.append(output_.text)
                res.append(temp)
            return res
        else:
            return [output.outputs[0].text for output in outputs]
