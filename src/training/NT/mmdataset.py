from typing import Dict, Any, List
import json
from datasets import Dataset
import pandas as pd
import os
from config import OmniTrainingConfig
from prompt import *
import numpy as np
import re
import random
def convert_probability(list_of_probability: List[float]) -> List[float]:
    num_list = [i/10 + 0.05 for i in range(10)]
    probability = np.array(list_of_probability) * np.array(num_list)

    return probability.sum()


def extract_answer_from_output(output_text: str) -> float:

    pattern = r'<answer>([^<]+)</answer>'
    matches = re.findall(pattern, output_text, re.IGNORECASE)
    
    if matches:
        answer_str = matches[-1].strip()
        try:
            value = float(answer_str)
            if value > 1.0:
                value = value / 100.0
            return max(0.0, min(1.0, value))
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def convert_probability_wikivideo(list_of_probability: List[str]) -> List[float]:
    lst = []
    for item in list_of_probability:
        if '<answer>' in item:

            lst.append(extract_answer_from_output(item))
        else:
            continue


    return np.mean(lst)

class PromptBuilder:
    @staticmethod

    def build_messages_vl(claim: str, video_path: str,config: OmniTrainingConfig) -> list[dict[str, any]]:
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
                        "text": PROMPT.format(text=claim),
                    },
                    {
                        "type": "video",
                        "video": video_path,
                        # "min_pixels": 32*32,
                        # "min_pixels": 256*256,
                        "resized_height": 256,
                        "resized_width": 256,
                        "fps": 0.5,
                    },
                ],
            }
        ]

    def build_messages_audio(claim: str, audio_path: str, config: OmniTrainingConfig) -> list[dict[str, any]]:
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
                        "text": PROMPT_AUDIO.format(text=claim),
                    },
                    {
                        "type": "audio",
                        "audio": audio_path,
                    },
                ],
            }
        ]

    def build_messages_text(claim: str, sentence: str, config: OmniTrainingConfig) -> list[dict[str, any]]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_TEXT,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT.format(sentence=sentence, claim=claim),
                    }
                ]
            }
        ]


class WikiVideoDataset:
    def __init__(self, config: OmniTrainingConfig, modality: str):
        self.config = config
        self.wikivideo_pre_path = config.wikivideo_pre_path
        self.wikivideo_label_path = config.wikivideo_label_path
        self.modality = modality

        if modality == 'omni':
            with open(config.omni_path, 'r') as f:
                self.data = json.load(f)
        if modality == 'video':
            with open(config.video_path, 'r') as f:
                self.data = json.load(f)

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        # train data
        train_data = []
        event_lst = self.config.wikivideo_eval_tag
        data = {k: v for k, v in self.data.items() if k not in event_lst}
        for e in data.values():
            for item in e:
                if self.modality == 'omni':
                    path = item['path']
                else:
                    path = item['path']

                message = {
                    'messages': PromptBuilder.build_messages_vl(item['claim'], path, self.config),
                    'answer': convert_probability_wikivideo(item['answer']),
                    'label': item['label'],
                    'modality': self.modality,
                }
                # 2 digits after the decimal point
                message['answer'] = round(message['answer'], 2)
                message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
                if (message['answer'] >= 0.5 and message['label'] == True) or (message['answer'] < 0.5 and message['label'] == False):
                    train_data.append(message)

        # eval data
        eval_data = []
        data = {k: v for k, v in self.data.items() if k in event_lst}
        for e in data.values():
            for item in e:
                if self.modality == 'omni':
                    path = item['path']
                else:
                    path = item['path']
                message = {
                    'messages': PromptBuilder.build_messages_vl(item['claim'], item['path'], self.config),
                    'answer': convert_probability(item['answer']),
                    'label': item['label'],
                    'modality': 'omni',
                }
                message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
                if (message['answer'] >= 0.5 and message['label'] == True) or (message['answer'] < 0.5 and message['label'] == False):
                    eval_data.append(message)

        True_label = 0
        False_label = 0
        True_item = []
        False_item = []
        for item in train_data:
            if item['label']:
                True_label += 1
                True_item.append(item)
            else:
                False_label += 1
                False_item.append(item)
        print(f'True_label: {True_label}')
        print(f'False_label: {False_label}')
        random.seed(42)
        if True_label < False_label:
            random.shuffle(False_item)
            if self.config.wikivideo_ratio == -1:
                pass
            else:
                False_item = False_item[:int(self.config.wikivideo_ratio*True_label)]
        else:
            random.shuffle(True_item)
            True_item = True_item[:False_label]
        train_data = True_item + False_item
        random.shuffle(train_data)
        
        return train_data, eval_data


class ClothoDataset:
    def __init__(self, config: OmniTrainingConfig):
        self.config = config
        self.clotho_pre_path = config.clotho_pre_path
        with open(config.audio_path, 'r') as f:
            self.data = json.load(f)

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        train_data = []
        eval_data = []
        for item in self.data['development']:
            path = item['path']
            message = {
                'messages': PromptBuilder.build_messages_audio(item['claim'], path, self.config),
                'answer': convert_probability(item['answer']),
                'label': item['label'],
                'modality': 'audio',
            }
            message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
            if (message['answer'] >= 0.5 and message['label'] == True) or (message['answer'] < 0.5 and message['label'] == False):
                train_data.append(message)
        for item in self.data['evaluation']:
            path = item['path']
            message = {
                'messages': PromptBuilder.build_messages_audio(item['claim'], path, self.config),
                'answer': convert_probability(item['answer']),
                'label': item['label'],
                'modality': 'audio',
            }
            message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
            eval_data.append(message)
        return train_data, eval_data


class UNLI_Dataset:
    def __init__(self, config: OmniTrainingConfig):
        self.config = config
        self.unli_label_path = config.unli_label_path
        self.processed_data = self.extract_data()

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        type_lst = ['test','train','validation']
        result = {}

        for t in type_lst:
            result[t] = []
            
            with open(os.path.join(self.unli_label_path, f'{t}.jsonl'), 'r') as f:
                for line in f:
                    item = json.loads(line)

                    result[t].append({
                        'sentence': item['premise'],
                        'claim': item['hypothesis'],
                        'type': t,
                        'path': os.path.join(self.unli_label_path, f'{t}.jsonl'),
                        'label': True if item['label'] >0.5 else False,
                        'probability': item['label'],
                        'modality': 'text',
                    })

        train_data = []
        eval_data = []
        for item in result['train']:
            message = {
                'messages': PromptBuilder.build_messages_text(item['claim'], item['sentence'], self.config),
                'answer': item['probability'],
                'label': item['label'],
                'modality': 'text',
            }
            message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
            if (message['answer'] >= 0.5 and message['label'] == True) or (message['answer'] < 0.5 and message['label'] == False):
                train_data.append(message)

        for item in result['validation']:
            message = {
                'messages': PromptBuilder.build_messages_text(item['claim'], item['sentence'], self.config),
                'answer': item['probability'],
                'label': item['label'],
                'modality': 'text',
            }
            message['messages'].append({"role": "assistant", "content": [{"type": "text", "text":'<answer>' + str(message['answer']) + '</answer>'}]})
            eval_data.append(message)
        return train_data, eval_data


class FakeDataset:
    def __init__(self, modality: str):

        self.modality = modality
    

    def construct_prompt(self, item: Dict[str, Any]) -> Dict[str, Any]:


        message = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": NLI_SYSTEM_PROMPT}]},

                {"role": "user", "content":
                [
                    {"type": "text", "text": NLI_PROMPT_TEXT.format(sentence=item['sentence'], text=item['claim'])},
                ]
                }
            ],
            "modality": item['modality'],
            "answer": item['label']
        }
        return message

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        result = []
        for i in range(1000):
            result.append(self.construct_prompt({
                'sentence': 'This is a fake sentence',
                'claim': 'This is a fake claim',
                'type': 'train',
                'path': 'fake_path',
                'label': True,
                'modality': self.modality,
            }))
        return result,result

