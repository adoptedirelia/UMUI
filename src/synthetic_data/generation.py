import os
import json
from typing import Dict, List, Any
from tqdm import tqdm
from .config import AppConfig
import torch
from .mmdataset import ClothoDataset,WikiVideoDataset,ViolinDataset,UNLI_Dataset,PeopleProfileDataset
import subprocess
import random
random.seed(42)
class SyntheticDataGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        # self.dataset_loader = DatasetLoader(config)
        # different dataset
        if config.dataset_name == 'wikivideo':
            self.dataset_loader = WikiVideoDataset(config)
        elif config.dataset_name == 'clotho':
            self.dataset_loader = ClothoDataset(config)
        elif config.dataset_name == 'violin':
            self.dataset_loader = ViolinDataset(config)
        elif config.dataset_name == 'unli':
            self.dataset_loader = UNLI_Dataset(config)
        elif config.dataset_name == 'peopleprofile':
            self.dataset_loader = PeopleProfileDataset(config)
        

        


        if config.generate_score:
            self.engine = self._create_score_engine()
        else:
            self.engine = self._create_engine()

    def _create_score_engine(self):
        if self.config.modality == 'omni':
            if 'qwen3' in self.config.model.lower():
                from .score_engine.Qwen3_omni import Qwen3_OmniHFInferenceEngine
                return Qwen3_OmniHFInferenceEngine(self.config)
            else:
                from .score_engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine
                return Qwen2_5_OmniHFInferenceEngine(self.config)
        elif self.config.modality == 'video':
            if 'qwen3' in self.config.model.lower():
                from .score_engine.Qwen3_vl import Qwen3_VLHFInferenceEngine

                return Qwen3_VLHFInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .score_engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine
                return Qwen2_5_OmniHFInferenceEngine(self.config)
            else:
                from .score_engine.Qwen2_5_vl import Qwen2_5_VLHFInferenceEngine
                return Qwen2_5_VLHFInferenceEngine(self.config)
        elif self.config.modality == 'audio':
            if 'qwen3' in self.config.model.lower():
                from .score_engine.Qwen3_omni import Qwen3_OmniHFInferenceEngine
                return Qwen3_OmniHFInferenceEngine(self.config)
            else:
                from .score_engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine
                return Qwen2_5_OmniHFInferenceEngine(self.config)

    def _create_engine(self):
        if self.config.modality == 'text':
            if 'qwen3' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen3_omni import Qwen3_OmniVLLMInferenceEngine
                return Qwen3_OmniVLLMInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine, Qwen2_5_OmniVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_OmniHFInferenceEngine(self.config)
                return Qwen2_5_OmniVLLMInferenceEngine(self.config)
            elif 'qwen3' in self.config.model.lower() and 'vl' in self.config.model.lower():
                from .engine.Qwen3_vl import Qwen3_VLHFInferenceEngine, Qwen3_VLVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen3_VLHFInferenceEngine(self.config)
                return Qwen3_VLVLLMInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'vl' in self.config.model.lower():
                from .engine.Qwen2_5_vl import Qwen2_5_VLHFInferenceEngine, Qwen2_5_VLVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_VLHFInferenceEngine(self.config)
                return Qwen2_5_VLVLLMInferenceEngine(self.config)
            elif 'qwen' in self.config.model.lower() and 'audio' in self.config.model.lower():
                from .engine.Qwen_audio import Qwen2AudioHFInferenceEngine, Qwen2AudioVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2AudioHFInferenceEngine(self.config)
                return Qwen2AudioVLLMInferenceEngine(self.config)
            elif 'flamingo' in self.config.model.lower():
                from .engine.audio_flamingo import AudioFlamingoInferenceEngine
                return AudioFlamingoInferenceEngine(self.config)
            elif 'vinci' in self.config.model.lower():
                from .engine.omni_vinci import OmniVinciInferenceEngine
                return OmniVinciInferenceEngine(self.config)
            elif 'qwen3' in self.config.model.lower():
                from .engine.Qwen3_text import Qwen3TextHFInferenceEngine, Qwen3TextVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen3TextHFInferenceEngine(self.config)
                return Qwen3TextVLLMInferenceEngine(self.config)
            else:
                from .engine.Qwen_text import QwenTextHFInferenceEngine, QwenTextVLLMInferenceEngine
                if self.config.backend == "hf":
                    return QwenTextHFInferenceEngine(self.config)
                return QwenTextVLLMInferenceEngine(self.config)
        elif self.config.modality == 'audio':
            if 'qwen3' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen3_omni import Qwen3_OmniVLLMInferenceEngine
                return Qwen3_OmniVLLMInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine, Qwen2_5_OmniVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_OmniHFInferenceEngine(self.config)
                return Qwen2_5_OmniVLLMInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'audio' in self.config.model.lower():
                from .engine.Qwen_audio import Qwen2AudioHFInferenceEngine, Qwen2AudioVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2AudioHFInferenceEngine(self.config)
                return Qwen2AudioVLLMInferenceEngine(self.config)
            elif 'flamingo' in self.config.model.lower():
                from .engine.audio_flamingo import AudioFlamingoInferenceEngine
                return AudioFlamingoInferenceEngine(self.config)
            elif 'vinci' in self.config.model.lower():
                from .engine.omni_vinci_audio import OmniVinciAudioInferenceEngine
                return OmniVinciAudioInferenceEngine(self.config)
            else:
                from .engine.Qwen_audio import Qwen2AudioHFInferenceEngine, Qwen2AudioVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2AudioHFInferenceEngine(self.config)
                return Qwen2AudioVLLMInferenceEngine(self.config)
        elif self.config.modality == 'omni':
            if 'qwen3' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen3_omni import Qwen3_OmniVLLMInferenceEngine
                return Qwen3_OmniVLLMInferenceEngine(self.config)
            elif 'qwen2.5' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine, Qwen2_5_OmniVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_OmniHFInferenceEngine(self.config)
                return Qwen2_5_OmniVLLMInferenceEngine(self.config)
            elif 'vinci' in self.config.model.lower():
                from .engine.omni_vinci import OmniVinciInferenceEngine
                return OmniVinciInferenceEngine(self.config)

        elif self.config.modality == 'video':
            if 'qwen3.5' in self.config.model.lower():
                from .engine.Qwen3_5 import Qwen3_5_VLVLLMInferenceEngine
                return Qwen3_5_VLVLLMInferenceEngine(self.config)
            if 'qwen3' in self.config.model.lower() and 'vl' in self.config.model.lower():
                from .engine.Qwen3_vl import Qwen3_VLHFInferenceEngine, Qwen3_VLVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen3_VLHFInferenceEngine(self.config)
                return Qwen3_VLVLLMInferenceEngine(self.config)
            elif 'qwen3' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen3_omni import Qwen3_OmniVLLMInferenceEngine
                return Qwen3_OmniVLLMInferenceEngine(self.config)

            elif 'qwen2.5' in self.config.model.lower() and 'omni' in self.config.model.lower():
                from .engine.Qwen_omni import Qwen2_5_OmniHFInferenceEngine, Qwen2_5_OmniVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_OmniHFInferenceEngine(self.config)
                return Qwen2_5_OmniVLLMInferenceEngine(self.config)
            elif 'vinci' in self.config.model.lower():
                from .engine.omni_vinci import OmniVinciInferenceEngine
                return OmniVinciInferenceEngine(self.config)
            else:
                from .engine.Qwen2_5_vl import Qwen2_5_VLHFInferenceEngine, Qwen2_5_VLVLLMInferenceEngine
                if self.config.backend == "hf":
                    return Qwen2_5_VLHFInferenceEngine(self.config)
                return Qwen2_5_VLVLLMInferenceEngine(self.config)


    def _create_eval_dataset(self):
        if self.config.dataset_name == 'wikivideo':
            selected_data = {k: v for k, v in self.dataset_loader.extract_data().items() if k in self.config.wikivideo_eval_tag}
            return selected_data
        elif self.config.dataset_name == 'clotho':
            selected_data = {k: v for k, v in self.dataset_loader.extract_data().items() if k in self.config.clotho_eval_tag}
            return selected_data
        elif self.config.dataset_name == 'violin':
            selected_data = {k: v for k, v in self.dataset_loader.extract_data().items() if k in self.config.violin_eval_tag}
            return selected_data
        elif self.config.dataset_name == 'unli':
            selected_data = {k: v for k, v in self.dataset_loader.extract_data().items() if k in self.config.unli_eval_tag}
            return selected_data
        elif self.config.dataset_name == 'peopleprofile':
            selected_data = {k: v for k, v in self.dataset_loader.extract_data().items() if k in self.config.peopleprofile_eval_tag}
            return selected_data


    def create_human_dataset(self):
        with open(self.config.human_data_path, 'r') as f:
            human_data = json.load(f)
        result = {}
        for item in human_data['dzhang98']:

            if item['event'] not in result:
                result[item['event']] = []
            result[item['event']].append({
                'claim': item['claim'],
                'path': os.path.join(self.config.wikivideo_pre_path, item['video']),
                'label': item['label'],
                'modality': 'video',
                'type': item['event'],
            })
        return result

    def run(self) -> List[Dict[str, Any]]:
        modality = self.config.modality
        if self.config.human_only:
            dataset = self.create_human_dataset()
        else:
            if self.config.evaluate:
                dataset = self._create_eval_dataset()
            else:
                if self.config.dataset_name == 'wikivideo':
                    dataset = {k: v for k, v in self.dataset_loader.extract_data().items() if k not in self.config.wikivideo_eval_tag}
                else:
                    dataset = self.dataset_loader.extract_data()

        batch_size = 1
        if modality == 'text':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist
            # if self.config.dataset_name == 'wikivideo':
                # unfold_data = filter_data_with_audio_track(unfold_data)
            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), batch_size)):
                batch_data = unfold_data[i:i+batch_size]
                answers = self.engine.generate_batch_text(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "sentence": item["sentence"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
            return results
        elif modality == 'audio':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist
            # if self.config.dataset_name == 'wikivideo':
                # unfold_data = filter_data_with_audio_track(unfold_data)
            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), batch_size)):
                batch_data = unfold_data[i:i+batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
            return results
        elif modality == 'video':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist

            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), batch_size)):
                batch_data = unfold_data[i:i+batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })

            return results
        elif modality == 'omni':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist

            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), batch_size)):
                batch_data = unfold_data[i:i+batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
            return results



    def run_batch(self) -> List[Dict[str, Any]]:
        modality = self.config.modality
        if self.config.human_only:
            dataset = self.create_human_dataset()
        else:
            if self.config.evaluate:
                dataset = self._create_eval_dataset()
            else:
                if self.config.dataset_name == 'wikivideo':
                    dataset = {k: v for k, v in self.dataset_loader.extract_data().items() if k not in self.config.wikivideo_eval_tag}
                else:
                    dataset = self.dataset_loader.extract_data()

        if modality == 'text':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist

            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), self.config.batch_size)):
                batch_data = unfold_data[i:i+self.config.batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
                    
            return results
        elif modality == 'audio':
            unfold_data = [item for sublist in dataset.values() for item in sublist]


            unfold_data = filter_data(unfold_data) # filter out the data that does not exist
            results: dict[str, List[Dict[str, Any]]] = {}
            for type in dataset.keys():
                results[type] = []
            for i in tqdm(range(0, len(unfold_data), self.config.batch_size)):
                batch_data = unfold_data[i:i+self.config.batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
            return results

        elif modality == 'video':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist

            # True_label = 0
            # False_label = 0
            # True_item = []
            # False_item = []
            # for item in unfold_data:
            #     if item['label']:
            #         True_label += 1
            #         True_item.append(item)
            #     else:
            #         False_label += 1
            #         False_item.append(item)
            # print(f'True_label: {True_label}')
            # print(f'False_label: {False_label}')
            # random.seed(42)
            # if True_label < False_label:
            #     random.shuffle(False_item)
            #     False_item = False_item[2*True_label:]
            # else:
            #     random.shuffle(True_item)
            #     True_item = True_item[:False_label]

            # unfold_data = False_item
            # random.shuffle(unfold_data)

            if self.config.data_split == -1:
                begin_index = 0
                end_index = len(unfold_data)
            else:
                begin_index = int(self.config.data_split/6*len(unfold_data))
                end_index = int((self.config.data_split+1)/6*len(unfold_data))
            unfold_data = unfold_data[begin_index:end_index]
            unfold_data = split_data_by_array_job(unfold_data, self.config.array_job_id, self.config.array_total_jobs)

            unfold_data, results = filter_data_by_array_job(unfold_data, self.config.output_path, dataset)
            

            # if os.path.exists(self.config.output_path):
            #     with open(self.config.output_path, "r") as f:
            #         results = json.load(f)
            #     for type in dataset.keys():
            #         if type not in results:
            #             results[type] = []
            #     # filter out the data that already exists, needs path and claim to be the same
            #     unfold_data2 = [item for sublist in results.values() for item in sublist]
            #     temp = []
            #     for j in unfold_data:
            #         flag = False
            #         for i in unfold_data2:
            #             if i["path"] == j["path"] and i["claim"] == j["claim"]:
            #                 flag = True
            #                 break
            #         if not flag:
            #             temp.append(j)
            #     unfold_data = temp
            # else:
            #     results: dict[str, List[Dict[str, Any]]] = {}
            #     for type in dataset.keys():
            #         results[type] = []

            for i in tqdm(range(0, len(unfold_data), self.config.batch_size)):
                batch_data = unfold_data[i:i+self.config.batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
                os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                with open(self.config.output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
            return results

        elif modality == 'omni':
            unfold_data = [item for sublist in dataset.values() for item in sublist]
            unfold_data = filter_data(unfold_data) # filter out the data that does not exist

            
            if self.config.data_split == -1:
                begin_index = 0
                end_index = len(unfold_data)
            else:
                begin_index = int(self.config.data_split/3*len(unfold_data))
                end_index = int((self.config.data_split+1)/3*len(unfold_data))
            unfold_data = unfold_data[begin_index:end_index]
            unfold_data = split_data_by_array_job(unfold_data, self.config.array_job_id, self.config.array_total_jobs)
            unfold_data, results = filter_data_by_array_job(unfold_data, self.config.output_path, dataset)
            # if os.path.exists(self.config.output_path):
            #     with open(self.config.output_path, "r") as f:
            #         results = json.load(f)
            #     for type in dataset.keys():
            #         if type not in results:
            #             results[type] = []
            #     # filter out the data that already exists, needs path and claim to be the same
            #     unfold_data2 = [item for sublist in results.values() for item in sublist]
            #     temp = []
            #     for j in unfold_data:
            #         flag = False
            #         for i in unfold_data2:
            #             if i["path"] == j["path"] and i["claim"] == j["claim"]:
            #                 flag = True
            #                 break
            #         if not flag:
            #             temp.append(j)
            #     unfold_data = temp
            # else:
            #     results: dict[str, List[Dict[str, Any]]] = {}
            #     for type in dataset.keys():
            #         results[type] = []
            
            for i in tqdm(range(0, len(unfold_data), self.config.batch_size)):
                batch_data = unfold_data[i:i+self.config.batch_size]
                answers = self.engine.generate_batch(
                    batch_data=batch_data,
                )
                for item, answer in zip(batch_data, answers):
                    results[item["type"]].append({
                        "claim": item["claim"],
                        "type": item["type"],
                        "path": item["path"],
                        "label": item["label"],
                        "modality": item["modality"],
                        "answer": answer,
                    })
                os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
                with open(self.config.output_path, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
            return results


        



def has_audio_track(video_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return True
    return False

def filter_data_with_audio_track(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in data if has_audio_track(item["path"])]

def filter_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in data if os.path.exists(item["path"])]

def split_data_by_array_job(data: List[Dict[str, Any]], array_job_id: int, array_total_jobs: int) -> List[Dict[str, Any]]:
    if array_total_jobs == -1:
        return data
    else:
        begin_index = int(array_job_id/array_total_jobs*len(data))
        end_index = int((array_job_id+1)/array_total_jobs*len(data))
        return data[begin_index:end_index]

def filter_data_by_array_job(data: List[Dict[str, Any]],output_path: str,dataset: dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if 'data_split' in output_path:
        temp_output_path = output_path.split('_data_split')[0] + '.json'
    else:
        temp_output_path = output_path

    if os.path.exists(temp_output_path):
        with open(temp_output_path, "r") as f:
            results = json.load(f)
        for type in dataset.keys():
            if type not in results:
                results[type] = []
        # filter out the data that already exists, needs path and claim to be the same
        unfold_data2 = [item for sublist in results.values() for item in sublist]
        temp = []
        already= []
        for j in data: 
            flag = False
            for i in unfold_data2:
                if i["path"] == j["path"] and i["claim"] == j["claim"]:
                    flag = True
                    break
            if not flag:
                temp.append(j)
            else:
                already.append(i)
        data = temp
        results2 = {}
        for type in dataset.keys():
            results2[type] = []
        for item in already:
            results2[item["type"]].append(item)
        return data, results2
    else:
        results: dict[str, List[Dict[str, Any]]] = {}
        for type in dataset.keys():
            results[type] = []

        return data, results

