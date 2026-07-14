from src.prompt_score import INSTRUCTION, PROMPT, SYSTEM_PROMPT
from src.synthetic_data.config import AppConfig
import torch
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



class Qwen3_VLHFInferenceEngine:
    def __init__(self, config: AppConfig) -> None:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.model, cache_dir=config.cache_dir, device_map="auto",torch_dtype="auto",attn_implementation="flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.processor = AutoProcessor.from_pretrained(config.model, cache_dir=config.cache_dir)
        self.score_token = ['0','1','2','3','4','5','6','7','8','9']

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        messages = PromptBuilder.build_messages_vl(claim, video_path, self.config)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors="pt")

        with torch.no_grad():
            generated_ids = self.model(**inputs,do_sample=True,temperature=0.7,top_p=0.95)

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
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            generated_ids = self.model(**inputs)

        logits = generated_ids.logits[:,-1,:]

        score_logits = logits[:,self.tokenizer.convert_tokens_to_ids(self.score_token)]
        score = torch.softmax(score_logits, dim=-1).tolist()
        return score