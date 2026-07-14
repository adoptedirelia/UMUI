from vllm import LLM, SamplingParams
import src.prompt_binary
import src.prompt
from src.synthetic_data.config import AppConfig


class PromptBuilder:
    @staticmethod
    def build_messages_text(claim: str, sentence: str, config: AppConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": 
                        src.prompt_binary.SYSTEM_PROMPT_TEXT if config.binary else src.prompt.SYSTEM_PROMPT_TEXT,
            },
            {
                "role": "user",
                "content": 
                        src.prompt_binary.INSTRUCTION_TEXT if config.binary else src.prompt.INSTRUCTION_TEXT + "\n\n" + (src.prompt_binary.PROMPT_TEXT if config.binary else src.prompt.PROMPT_TEXT).format(sentence=sentence, claim=claim),
            }
        ]

class Qwen3TextHFInferenceEngine: # text only
    def __init__(self, config: AppConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.modality = config.modality
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.model, cache_dir=config.cache_dir, device_map="auto",enable_thinking=False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir, padding_side="left")

    def generate(self, batch_data: list[dict[str, any]]) -> str:
        claim = [item["claim"] for item in batch_data][0]
        sentence = [item["sentence"] for item in batch_data][0]
        messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return outputs[0]

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        outputs = []
        claims = [item["claim"] for item in batch_data]
        sentences = [item["sentence"] for item in batch_data]
        messages = []
        for claim, sentence in zip(claims, sentences):
            message = PromptBuilder.build_messages_text(claim, sentence, self.config)
            messages.append(message)
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        return outputs

class Qwen3TextVLLMInferenceEngine: # text only
    def __init__(self, config: AppConfig) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.cache_dir,enable_thinking=False)
        self.llm = LLM(
            model=config.model,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
        )
    
    def generate(self, batch_data: list[dict[str, any]]) -> str:
        claim = [item["claim"] for item in batch_data][0]
        sentence = [item["sentence"] for item in batch_data][0]
        messages = PromptBuilder.build_messages_text(claim, sentence, self.config)
        outputs = self.llm.generate(messages, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens))
        return outputs[0].outputs[0].text

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:

        claims = [item["claim"] for item in batch_data]
        sentences = [item["sentence"] for item in batch_data]
        messages = []
        for claim, sentence in zip(claims, sentences):
            message = PromptBuilder.build_messages_text(claim, sentence, self.config)
            messages.append(self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True,enable_thinking=False))
        out = self.llm.generate(messages, sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens))
        outputs = [output.outputs[0].text for output in out]
        return outputs