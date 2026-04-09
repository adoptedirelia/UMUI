from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

from src.synthetic_data.config import AppConfig
import src.prompt_binary
import src.prompt

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

class OmniVinciInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        self.djconfig = config
        self.modality = config.modality
        self.generation_kwargs = {"max_new_tokens": self.djconfig.max_new_tokens, "max_length": 1024}

        self.num_video_frames = 32
        self.audio_length = "max_600"
        self.config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.model, trust_remote_code=True, torch_dtype="torch.float16", device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        self.generation_config = self.model.default_generation_config

        self.generation_config.update(**self.generation_kwargs)
        if config.modality == 'omni':
            self.load_audio_in_video = True
        else:
            self.load_audio_in_video = False
            
        self.model.config.load_audio_in_video = self.load_audio_in_video
        self.processor.config.load_audio_in_video = self.load_audio_in_video
        if self.num_video_frames > 0:
            self.model.config.num_video_frames = self.num_video_frames
            self.processor.config.num_video_frames = self.num_video_frames


        if self.audio_length != -1:
            self.model.config.audio_chunk_length = self.audio_length
            self.processor.config.audio_chunk_length = self.audio_length


    def generate(self, claim: str, path: str) -> str:
        messages = PromptBuilder.build_messages_vl(claim, path)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor([text])


        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                media=getattr(inputs, 'media', None),
                media_config=getattr(inputs, 'media_config', None),
                generation_config=self.generation_config,
            )
        decoded_outputs = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded_outputs[0]

    def _generate_single(self, messages: list[dict]) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor([text])
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                media=getattr(inputs, 'media', None),
                media_config=getattr(inputs, 'media_config', None),
                generation_config=self.generation_config,
            )
        decoded_outputs = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded_outputs[0]

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        # omni 模式批量推理时模型内部音频索引会越界，改为逐条推理
        if self.modality == 'omni':
            outputs = []
            for item in batch_data:
                messages = PromptBuilder.build_messages_vl(item["claim"], item["path"], self.djconfig)
                outputs.append(self._generate_single(messages))
            return outputs

        inputs_lst = []
        if self.modality == 'text':
            for item in batch_data:
                messages = PromptBuilder.build_messages_text(item["claim"], item["sentence"], self.djconfig)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs_lst.append(text)
        elif self.modality == 'video':
            for item in batch_data:
                messages = PromptBuilder.build_messages_vl(item["claim"], item["path"], self.djconfig)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs_lst.append(text)

        inputs = self.processor(inputs_lst)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                media=getattr(inputs, 'media', None),
                media_config=getattr(inputs, 'media_config', None),
                generation_config=self.generation_config,
            )
        decoded_outputs = self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded_outputs