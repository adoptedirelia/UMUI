
from pickle import FALSE
from typing import Dict, Any, List
import json
from datasets import Dataset
import pandas as pd
import os
import random

from config import OmniEvalConfig

class WikiVideoDataset:
    def __init__(self, config: OmniEvalConfig):
        self.modality = config.modality
        self.wikivideo_pre_path = config.wikivideo_pre_path
        self.wikivideo_label_path = config.wikivideo_label_path
        self.wikivideo_audio_pre_path = config.wikivideo_audio_pre_path
        with open(self.wikivideo_label_path, 'r') as f:
            self.data = json.load(f)

        self.eval_tag = config.wikivideo_eval_tag

        self.processed_data = self.extract_data()
        # print(self.modality)

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        for event_name, samples in self.data.items():
            results[event_name] = []
            event_claims = samples['claims']
            event_video_paths = samples['claims_to_supporting_videos']
            event_sentences = samples['original_article']
            event_video_paths_list = list(samples['videos'].keys())
            seen = set()
            for sentence, claims in zip(event_sentences, event_claims):

                temp_dic = self.get_information(sentence,claims,event_video_paths,self.modality,event_name,event_video_paths_list)
                for item in temp_dic:
                    key = (item['path'], item['claim'])
                    if key not in seen:
                        seen.add(key)
                        results[event_name].append(item)
        new_results = {}
        for event in self.eval_tag:
            new_results[event] = results[event]
        return new_results
    
    def get_information(self,sentence:str,claims:list[str],video_paths:list[str],modality:str,event_name:str,event_video_paths_list:list[str]) -> Dict[str, Any]:

        result = []
        for claim in claims:
            
            video_path = video_paths[claim]['videos_modalities']

            for video in event_video_paths_list:
                if not os.path.exists(os.path.join(self.wikivideo_pre_path, video+'.mp4')):
                    continue
                # if video in video_path.keys():
                #     continue
                # else:
                #     result.append({
                #         'sentence': sentence,
                #         'claim': claim,
                #         'type': event_name,
                #         'path': os.path.join(self.wikivideo_pre_path, video+'.mp4'),
                #         'label': False,
                #         'modality': modality,
                #         })
            for video,label in video_path.items():
                if not os.path.exists(os.path.join(self.wikivideo_pre_path, video+'.mp4')):
                    continue
                
                if modality == 'video': 
                    gt_ = (label['video'] or label['ocr'])
                    result.append({
                        'sentence': sentence,
                        'claim': claim,
                        'type': event_name,
                        'path': os.path.join(self.wikivideo_pre_path, video+'.mp4'),
                        'label': gt_,
                        'modality': modality,
                        })
                elif modality == 'audio':
                    if not os.path.exists(os.path.join(self.wikivideo_audio_pre_path, video+'.wav')):
                        continue
                    gt_ = label['audio'] and not label['video'] and not label['ocr']
                    result.append({
                        'sentence': sentence,
                        'claim': claim,
                        'type': event_name,
                        'path': os.path.join(self.wikivideo_audio_pre_path, video+'.wav'),
                        'label': gt_,
                        'modality': modality,
                    })
                elif modality == 'omni':
                    gt_ = label['video'] or label['audio']
                    result.append({
                        'sentence': sentence,
                        'claim': claim,
                        'type': event_name,
                        'path': os.path.join(self.wikivideo_pre_path, video+'.mp4'),
                        'label': gt_,
                        'modality': modality,
                    })
        return result


# 
class ClothoDataset:
    def __init__(self, config: OmniEvalConfig):
        self.clotho_label_path = config.clotho_label_path
        self.audio_pre_path = config.clotho_pre_path
        self.processed_data = self.extract_data()


    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        type_lst = ['validation','evaluation','development']

        final_lst = {}
        for t in type_lst:
            final_lst[t] = []
            df = pd.read_csv(os.path.join(self.clotho_label_path, f'clotho_entailment_{t}.csv'))
            for i in df.iterrows():
                audio_file = os.path.join(self.audio_pre_path,t, i[1]['Audio file'])
                if not os.path.exists(audio_file):
                    continue
                entailment = i[1]['Entailment']
                contradiction = i[1]['Contradiction']

                final_lst[t].append({
                    'claim': entailment,
                    'type': t,
                    'path': audio_file,
                    'label': True,
                    'modality': 'audio',
                })
                final_lst[t].append({
                    'claim': contradiction,
                    'type': t,
                    'path': audio_file,
                    'label': False,
                    'modality': 'audio',
                })
        return final_lst['validation']

class UNLI_Dataset:
    def __init__(self, config: OmniEvalConfig):
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
        return result['validation']

class PeopleProfileDataset:
    def __init__(self, config: OmniEvalConfig):
        self.peopleprofile_label_path = config.peopleprofile_label_path
        self.processed_data = self.extract_data()

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        type_lst = ['train','test','dev']
        folder_lst = ['body','lead']
        result = {}
        for t in type_lst:
            result[t] = []
        for folder in folder_lst:
            for t in type_lst:
                with open(os.path.join(self.peopleprofile_label_path, folder, f'{t}.jsonl'), 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        score = item['score']
                        claim = item['subclaim-decontext']
                        sentence = item['snippet-sents']
                        if score < 0:
                            score = 0
                        for s in sentence:
                            result[t].append({
                                'sentence': s,
                                'claim': claim,
                                'type': t,
                                'path': os.path.join(self.peopleprofile_label_path, folder, f'{t}.jsonl'),
                                'label': True if score >0.5 else False,
                                'probability': score,
                                'modality': 'text',
                            })
        # make the True and False number same
        for t in result.keys():
            random.seed(42)
            random.shuffle(result[t])
            True_idx = [i for i in range(len(result[t])) if result[t][i]['label']]
            False_idx = [i for i in range(len(result[t])) if not result[t][i]['label']]
            if len(True_idx) > len(False_idx):
                result[t] = [result[t][i] for i in True_idx[:len(False_idx)]] + [result[t][i] for i in False_idx]
            else:
                result[t] = [result[t][i] for i in False_idx[:len(True_idx)]] + [result[t][i] for i in True_idx]

        return result['dev']


class ViolinDataset:
    def __init__(self, config: OmniEvalConfig):
        self.modality = config.modality
        self.violin_label_path = config.violin_label_path
        self.violin_pre_path = config.violin_pre_path
        self.processed_data = self.extract_data()

    def extract_data(self) -> Dict[str, List[Dict[str, Any]]]:
        with open(self.violin_label_path, 'r') as f:
            data = json.load(f)
        split = ['train', 'test', 'validate']
        result = {}
        for s in split:
            result[s] = []
        for key, value in data.items():
            if not os.path.exists(os.path.join(self.violin_pre_path, value['file']+'.mp4')):
                continue

            for claim in value['statement']:
                result[value['split']].append({
                    'claim': claim[0],
                    'type': value['split'],
                    'path': os.path.join(self.violin_pre_path, value['file']+'.mp4'),
                    'label': True,
                    'span': value['span'],
                    'modality': self.modality,
                })

                result[value['split']].append({
                    'claim': claim[1],
                    'type': value['split'],
                    'path': os.path.join(self.violin_pre_path, value['file']+'.mp4'),
                    'label': False,
                    'span': value['span'],
                    'modality': self.modality,
                })

        return result['validate']


def data_distribution(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:

    for t in data.keys():
        print(t)
        True_n = 0
        False_n = 0
        for i in data[t]:
            if i['label']:
                True_n += 1
            else:
                False_n += 1
        print(f'True_n: {True_n}')
        print(f'False_n: {False_n}')

