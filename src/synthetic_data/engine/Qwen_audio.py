from vllm import LLM, SamplingParams
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig

import librosa
import warnings
import os
warnings.filterwarnings("ignore")

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


class Qwen2AudioHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoTokenizer
        self.config = config
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(config.model, cache_dir=config.cache_dir, device_map="auto", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir, trust_remote_code=True)

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        claim = [item["claim"] for item in batch_data][0]
        path = [item["path"] for item in batch_data][0]
        prompt = PromptBuilder.build_messages_audio(claim, path, self.config)
        prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        audio, sr = librosa.load(path, sr=16000)

        inputs = self.processor(text=prompt, audios=audio, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response[0] if response else ""
    
    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        outputs = []
        claims = [item["claim"] for item in batch_data]
        paths = [item["path"] for item in batch_data]
        prompts = []
        audios = []
        for claim, path in zip(claims, paths):
            prompt = PromptBuilder.build_messages_audio(claim, path, self.config)
            prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            audio, sr = librosa.load(path, sr=16000)
            prompts.append(prompt)
            audios.append(audio)

        inputs = self.processor(text=prompts, audios=audios, return_tensors="pt",padding=True).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(responses)
        return responses

class Qwen2AudioVLLMInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from vllm import LLM, SamplingParams    
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        self.config = config
        self.modality = config.modality
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir, trust_remote_code=True)
        self.llm = LLM(
            model=config.model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 1},
            disable_custom_all_reduce=True,

        )
    def generate(self, batch_data: list[dict[str, any]]) -> str:
        if self.modality == 'audio':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            for claim, path in zip(claim, path):
                if os.path.exists(path):
                    audio, sr = librosa.load(path, sr=16000)
                    audio = audio[:480000]  # 截断到 30 秒，避免 mel 帧数超过 3000
                else:
                    continue
                prompt_dict = {
                "prompt": prompt,
                "multi_modal_data": {
                "audio": audio
                }}
        elif self.modality == 'text':
            sentence = [item["sentence"] for item in batch_data]
            claim = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentence, claim):
                prompt = PromptBuilder.build_messages_text(claim, sentence, self.config)
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_dict = {
                "prompt": prompt,
                "multi_modal_data": {
                }}
        outputs = self.llm.generate([prompt_dict], sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens))
        return [output.outputs[0].text for output in outputs]

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        llm_inputs = []
        if self.modality == 'audio':
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):
                if os.path.exists(path):
                    audio, sr = librosa.load(path, sr=16000)
                    audio = audio[:480000]  # 截断到 30 秒，避免 mel 帧数超过 3000
                else:
                    continue
                prompt = PromptBuilder.build_messages_audio(claim, path, self.config)
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                prompt_dict = {
                "prompt": prompt,
                "multi_modal_data": {
                "audio": audio
                }}
                llm_inputs.append(prompt_dict)
        elif self.modality == 'text':
            sentences = [item["sentence"] for item in batch_data]
            claims = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentences, claims):
                prompt = PromptBuilder.build_messages_text(claim, sentence, self.config)
                prompt = self.processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                llm_inputs.append(prompt)
                prompt_dict = {
                "prompt": prompt,
                "multi_modal_data": {
                }}
                llm_inputs.append(prompt_dict)

        outputs = self.llm.generate(llm_inputs, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens))
        return [output.outputs[0].text for output in outputs]