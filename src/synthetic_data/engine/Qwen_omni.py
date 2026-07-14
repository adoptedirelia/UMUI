from vllm import LLM, SamplingParams
# from src.prompt import INSTRUCTION, PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_AUDIO, INSTRUCTION_AUDIO, PROMPT_AUDIO, SYSTEM_PROMPT_TEXT, INSTRUCTION_TEXT, PROMPT_TEXT
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig
import warnings
import os
os.environ['VLLM_USE_V1'] = '0'
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
                        "text": src.prompt_binary.SYSTEM_PROMPT_AUDIO if config.binary else src.prompt.SYSTEM_PROMPT_AUDIO,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": src.prompt_binary.INSTRUCTION_AUDIO if config.binary else src.prompt.INSTRUCTION_AUDIO + "\n\n" + (src.prompt_binary.PROMPT_AUDIO if config.binary else src.prompt.PROMPT_AUDIO).format(text=claim),
                    },
                    {
                        "type": "audio",
                        "audio": audio_path,
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

    def build_raw_text(claim: str, config: AppConfig) -> str: # this if for qwen omni
        prompt = src.prompt_binary.INSTRUCTION if config.binary else src.prompt.INSTRUCTION + '\n\n' + (src.prompt_binary.PROMPT if config.binary else src.prompt.PROMPT).format(text=claim)
        text = (
        f"<|im_start|>system\n{src.prompt_binary.SYSTEM_PROMPT if config.binary else src.prompt.SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>\n"
        f"{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        )
        return text 



class Qwen2_5_OmniVLLMInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        self.modality = config.modality
        self.processor = Qwen2_5OmniProcessor.from_pretrained(config.model)
        self.process_mm_info = process_mm_info
        self.config = config
        if self.modality == 'video':
            self.limit_mm_per_prompt = {"video": 1}
        elif self.modality == 'audio':
            self.limit_mm_per_prompt = {"audio": 1}
        elif self.modality == 'omni':
            self.limit_mm_per_prompt = {"video": 1,"audio": 1}
        elif self.modality == 'text':
            self.limit_mm_per_prompt = {}

        self.llm = LLM(
            model=config.model,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            enable_prefix_caching=True,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_custom_all_reduce=True,
        )
    
    def process_data(self, raw_text: str, messages: list[dict[str, any]]) -> tuple[list[dict[str, any]], str]:
        if self.modality == 'video':
            USE_AUDIO_IN_VIDEO = False
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        elif self.modality == 'audio':
            USE_AUDIO_IN_VIDEO = True
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        elif self.modality == 'omni':
            try:
                USE_AUDIO_IN_VIDEO = True
                audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            except:
                USE_AUDIO_IN_VIDEO = False
                audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        elif self.modality == 'text':
            USE_AUDIO_IN_VIDEO = False
            audios, images, videos = None, None, None

        mm_data = {}
        if images is not None:
            mm_data["image"] = images
        if videos is not None:
            mm_data["video"] = videos
        if audios is not None:
            mm_data["audio"] = audios
        llm_inputs = {
            "prompt": raw_text,
            "multi_modal_data": mm_data,

            "mm_processor_kwargs": {
                "use_audio_in_video": USE_AUDIO_IN_VIDEO,
            },
        }
        return llm_inputs

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        if self.modality == 'video':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            messages = PromptBuilder.build_messages_vl(claim, path, self.config)
        elif self.modality == 'audio':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            messages = PromptBuilder.build_messages_audio(claim, path, self.config)
        elif self.modality == 'omni':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            messages = PromptBuilder.build_messages_vl(claim, path, self.config)
        elif self.modality == 'text':
            sentence = [item["sentence"] for item in batch_data]
            claim = [item["claim"] for item in batch_data]
            messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
        raw_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        llm_inputs = self.process_data(raw_text, messages)

        outputs = self.llm.generate([llm_inputs], sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p))
        return outputs[0].outputs[0].text

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:

        llm_inputs = []
        if self.modality in ['video', 'omni','audio']:
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):
                if self.modality == 'audio':
                    messages = PromptBuilder.build_messages_audio(claim, path, self.config)
                else:
                    messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                raw_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                llm_input = self.process_data(raw_text, messages)
                llm_inputs.append(llm_input)
        elif self.modality == 'text':
            sentences = [item["sentence"] for item in batch_data]
            claims = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentences, claims):
                messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
                raw_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                llm_input = self.process_data(raw_text, messages)
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


class Qwen2_5_OmniHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer
        from qwen_omni_utils import process_mm_info
        self.process_mm_info = process_mm_info

        self.config = config
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            config.model, cache_dir=config.cache_dir, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)

    
    def generate(self, claim: str, video_path: str, max_new_tokens: int) -> str:
        messages = PromptBuilder.build_messages_vl(claim, video_path, self.config)
        try:
            USE_AUDIO_IN_VIDEO = True
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(self.model.device).to(self.model.dtype)


            text_ids, audio = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

            outputs = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except:
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids = self.model.generate(**inputs)
            outputs = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs[0] if outputs else ""
    
    def generate_batch(self, claims: list[str], video_paths: list[str], max_new_tokens: int) -> list[str]:
        outputs = []
        for claim, video_path in zip(claims, video_paths):
            outputs.append(self.generate(claim, video_path, max_new_tokens))
        return outputs



