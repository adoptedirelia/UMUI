from src.prompt_score import INSTRUCTION, PROMPT, SYSTEM_PROMPT
import torch
from src.synthetic_data.config import AppConfig



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


class Qwen2_5_VLHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self.process_vision_info = process_vision_info
        self.config = config
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model, cache_dir=config.cache_dir, device_map="auto",torch_dtype="auto",attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.score_token = ['0','1','2','3','4','5','6','7','8','9']

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        claim = [item['claim'] for item in batch][0]
        video_path = [item['video_path'] for item in batch][0]
        messages = PromptBuilder.build_messages_vl(claim, video_path, self.config)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            generated_ids = self.model(**inputs)


        logits = generated_ids.logits[0,-1,:]
        score_logits = logits[self.tokenizer.convert_tokens_to_ids(self.score_token)]
        score = torch.softmax(score_logits, dim=-1)

        return score

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        outputs = []
        claims = [item['claim'] for item in batch_data]
        paths = [item['path'] for item in batch_data]
        messages = []
        for claim, path in zip(claims, paths):
            messages.append(PromptBuilder.build_messages_vl(claim, path, self.config))

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model(**inputs)
        logits = generated_ids.logits[:,-1,:]

        score_logits = logits[:,self.tokenizer.convert_tokens_to_ids(self.score_token)]
        score = torch.softmax(score_logits, dim=-1).tolist()

        return score
