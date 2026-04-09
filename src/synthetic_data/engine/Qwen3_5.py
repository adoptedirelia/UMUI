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


class Qwen3_5_VLVLLMInferenceEngine:
    def __init__(self, config: dict[str, any]) -> None:

        self.config = config
        self.llm = LLM(
            model=config.model,
            limit_mm_per_prompt={"video": 1},
            max_model_len=32768,
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=True,
            enable_prefix_caching=True,
            allowed_local_media_path='/'
        )


    def generate(self, batch_data: list[dict[str, any]]) -> str:
        if self.config.modality == 'video':
            video_path = "file://" + batch_data['path']
            video_content = {"type": "video_url", "video_url": {"url": video_path}}
            text_content = {"type": "text", "text": src.prompt.INSTRUCTION + "\n\n" + src.prompt.PROMPT.format(text=batch_data['claim'])}
            content = [video_content, text_content]

        elif self.config.modality == 'text':
            text_content = {"type": "text", "text": batch_data['claim']}
            content = [text_content, text_content]

        outputs = self.llm.chat(
            [{"role": "user", "content": content}],
            sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p),
            mm_processor_kwargs={'fps': 0.5,'resized_width': 256,'resized_height': 256},
            chat_template_kwargs={"enable_thinking": True},
        )
        return outputs[0].outputs[0].text

    def generate_batch(self, batch_data: list[dict[str, any]]) -> list[str]:
        llm_inputs = []
        if self.config.modality == 'video':
            claims = [item["claim"] for item in batch_data]
            paths = [item["path"] for item in batch_data]
            for claim, path in zip(claims, paths):
                video_path = "file://" + path
                video_content = {"type": "video_url", "video_url": {"url": video_path}}
                text_content = {"type": "text", "text": src.prompt.INSTRUCTION + "\n\n" + src.prompt.PROMPT.format(text=claim)}
                content = [video_content, text_content]
                llm_inputs.append([{"role": "user", "content": content}])
        elif self.config.modality == 'text':
            sentences = [item["sentence"] for item in batch_data]
            claims = [item["claim"] for item in batch_data]
            for sentence, claim in zip(sentences, claims):
                text_content = {"type": "text", "text": sentence}
                content = [text_content, text_content]
                llm_inputs.append([{"role": "user", "content": content}])

        outputs = self.llm.chat(
            llm_inputs,
            sampling_params=SamplingParams(max_tokens=self.config.max_new_tokens,n=self.config.response_num,temperature=self.config.temperature,top_p=self.config.top_p),
            mm_processor_kwargs={'fps': 0.5,'resized_width': 256,'resized_height': 256},
            chat_template_kwargs={"enable_thinking": True},
        )
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