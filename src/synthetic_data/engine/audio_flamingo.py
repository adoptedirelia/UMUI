from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig
import torch
import os
class PromptBuilder:
    @staticmethod
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
                        "text": src.prompt_binary.INSTRUCTION_AUDIO if config.binary else src.prompt.INSTRUCTION_AUDIO + "\n\n" + src.prompt.PROMPT_AUDIO.format(text=claim),
                    },
                    {
                        "type": "audio",
                        "path": audio_path,
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

class AudioFlamingoInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(config.model, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(config.model)
        self.config = config

    # def generate(self, claim: str, path: str, max_new_tokens: int) -> str:
    #     messages = PromptBuilder.build_messages_audio(claim, path)
    #     inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
    #     with torch.no_grad():
    #         outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    #     decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #     return decoded_outputs[0]


    # def generate_batch(self, claims: list[str], paths: list[str], max_new_tokens: int) -> list[str]:
    #     outputs = []
    #     input_list = []
    #     for claim, path in zip(claims, paths):

    #         if os.path.exists(path):
    #             input_list.append(PromptBuilder.build_messages_audio(claim, path))
            
    #     inputs = self.processor.apply_chat_template(input_list, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
    #     outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    #     decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    #     outputs = decoded_outputs

    #     return outputs

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        if self.config.modality == 'audio':
            claim = [item["claim"] for item in batch_data]
            path = [item["path"] for item in batch_data]
            messages = PromptBuilder.build_messages_audio(claim, path, self.config)
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
            decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return decoded_outputs[0]
        elif self.config.modality == 'text':
            sentence = [item["sentence"] for item in batch_data]
            claim = [item["claim"] for item in batch_data]
            messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
            decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return decoded_outputs[0]


    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        outputs = []
        input_list = []
        if self.config.modality == 'audio':
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):

                if os.path.exists(path):
                    input_list.append(PromptBuilder.build_messages_audio(claim, path, self.config))
                
            inputs = self.processor.apply_chat_template(input_list, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
                decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                outputs = decoded_outputs
        elif self.config.modality == 'text':
            sentences = [item["sentence"] for item in batch_data]
            claims = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentences, claims):
                input_list.append(PromptBuilder.build_messages_text(claim, sentence, self.config))
            inputs = self.processor.apply_chat_template(input_list, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
                decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                outputs = decoded_outputs
        return outputs