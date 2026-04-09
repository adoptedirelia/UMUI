from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

from src.synthetic_data.config import AppConfig
from src.prompt import SYSTEM_PROMPT_AUDIO, INSTRUCTION_AUDIO, PROMPT_AUDIO

class PromptBuilder:
    @staticmethod
    def build_messages_audio(claim: str, audio_path: str) -> list[dict[str, any]]:
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



class OmniVinciAudioInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        self.generation_kwargs = {"max_new_tokens": 128, "max_length": 1024}
        self.load_audio_in_video = True
        self.num_video_frames = 128
        self.audio_length = "max_3600"
        self.config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.model, trust_remote_code=True, torch_dtype="torch.float16", device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
        self.generation_config = self.model.default_generation_config
        self.generation_config.update(**self.generation_kwargs)
        self.model.config.load_audio_in_video = self.load_audio_in_video
        self.processor.config.load_audio_in_video = self.load_audio_in_video
        if self.num_video_frames > 0:
            self.model.config.num_video_frames = self.num_video_frames
            self.processor.config.num_video_frames = self.num_video_frames
        if self.audio_length != -1:
            self.model.config.audio_chunk_length = self.audio_length
            self.processor.config.audio_chunk_length = self.audio_length


    def generate(self, claim: str, path: str, max_new_tokens: int) -> str:
        messages = PromptBuilder.build_messages_audio(claim, path)
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
        outputs = []
        for item in batch_data:
            outputs.append(self.generate(item["claim"], item["path"], self.generation_kwargs["max_new_tokens"]))
        return outputs