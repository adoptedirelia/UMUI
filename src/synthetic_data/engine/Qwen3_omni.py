from vllm import LLM, SamplingParams
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig


import os
os.environ['VLLM_USE_V1'] = '0'
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



class Qwen3_OmniVLLMInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen3OmniMoeProcessor
        self.modality = config.modality
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(config.model)

        self.process_mm_info = process_mm_info
        self.config = config
        if self.modality == 'video':
            self.limit_mm_per_prompt = {"video": 1}
        elif self.modality == 'audio':
            self.limit_mm_per_prompt = {"audio": 1}
        elif self.modality == 'omni':
            self.limit_mm_per_prompt = {"video": 1,"audio": 1}
        else:
            self.limit_mm_per_prompt = {}

        self.llm = LLM(
            model=config.model,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            max_model_len=16384,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )
    
    def process_data(self,raw_text: str, messages: list[dict[str, any]]) -> tuple[list[dict[str, any]], str]:
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

    def process_data_omni(self,raw_text: str, messages: list[dict[str, any]]) -> tuple[list[dict[str, any]], str]:

        USE_AUDIO_IN_VIDEO = False
        audios, images, videos = self.process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
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

                if self.modality == 'video':
                    messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                elif self.modality == 'audio':
                    messages = PromptBuilder.build_messages_audio(claim, path)
                elif self.modality == 'omni':
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
        try:
            outputs = self.llm.generate(llm_inputs, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p))
        except Exception as e:
            llm_inputs = []
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):
                messages = PromptBuilder.build_messages_vl(claim, path, self.config)
                raw_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                llm_input = self.process_data_omni(raw_text, messages)
                llm_inputs.append(llm_input)
            outputs = self.llm.generate(llm_inputs, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p))
            
        if self.config.response_num > 1:
            res = []
            for output in outputs:
                temp = []
                for output_ in output.outputs:
                    temp.append(output_.text)
                res.append(temp)
            print(res)
            return res
        else:

            return [output.outputs[0].text for output in outputs]

