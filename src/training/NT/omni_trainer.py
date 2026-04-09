from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
from config import OmniTrainingConfig
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Sampler
from mmdataset import *
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="System prompt modified")

def find_start_idx(list_ids: List[int], ids: List[int]) -> int:
    """Find start index"""
    for i in range(len(list_ids)):
        if list_ids[i:i+len(ids)] == ids:
            return i
    return -1

class CollateFn:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        '''
        batch: List[Dict[str, Any]]
        batch = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What is the capital of France?'},
                ]
            }
        ]
        '''

        texts = [message['messages'] for message in batch]
        try:
            USE_AUDIO_IN_VIDEO = True
            text = self.processor.apply_chat_template(texts, add_generation_prompt=False, tokenize=False)
            audios, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audios, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs

        except Exception as e:
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(texts, add_generation_prompt=False, tokenize=False)
            audios, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=None, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs
        
        labels = inputs["input_ids"].clone()
        attention_mask = inputs.get("attention_mask", None)
        
        for i in range(len(inputs["input_ids"])):
            start = '<|im_start|>assistant\n'
            ids = self.processor.tokenizer.encode(start)
            inputs_ids = inputs["input_ids"][i]
            start_idx = find_start_idx(inputs_ids.tolist(), ids)
            
            if start_idx == -1:
                labels[i][:] = -100
                print(f"Warning: Could not find assistant start token in sample {i}")
            else:
                start_idx = start_idx + len(ids)
                labels[i][:start_idx] = -100
                labels[i][start_idx:] = inputs_ids[start_idx:]
            

            if attention_mask is not None:

                labels[i][attention_mask[i] == 0] = -100

        inputs["labels"] = labels
        inputs['use_audio_in_video'] = USE_AUDIO_IN_VIDEO
        return inputs

class DistributedModalityBatchSampler(Sampler):
    def __init__(self, modalities, batch_size, num_replicas=None, rank=None, shuffle=True, drop_last=False):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
            
        self.modalities = np.array(modalities)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = np.arange(len(self.modalities))

        modality_to_indices = {}
        for mod in np.unique(self.modalities):
            mod_idx = indices[self.modalities == mod]
            if self.shuffle:

                mod_idx = mod_idx[torch.randperm(len(mod_idx), generator=g).tolist()]

            if len(mod_idx) % self.num_replicas != 0:
                padding_size = self.num_replicas - (len(mod_idx) % self.num_replicas)
                mod_idx = np.concatenate([mod_idx, mod_idx[:padding_size]])
            modality_to_indices[mod] = mod_idx.tolist()


        final_batches = []
        for mod, idxs in modality_to_indices.items():

            rank_idxs = idxs[self.rank::self.num_replicas]
            # different modalities can use different batch sizes
            for i in range(0, len(rank_idxs), self.batch_size):
                batch = rank_idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                if len(batch) > 0:
                    final_batches.append(batch)

        if self.shuffle:
            rand_idx = torch.randperm(len(final_batches), generator=g).tolist()
            final_batches = [final_batches[i] for i in rand_idx]

        return iter(final_batches)

    def __len__(self):
        return len(self.modalities) // (self.batch_size * self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, va_ds, video_ds, audio_ds, text_ds):
        self.data = []
        self.modalities = []
        
        if va_ds is not None:
            for item in va_ds:
                self.data.append(item)

                self.modalities.append("omni")

        if video_ds is not None:
            for item in video_ds:
                self.data.append(item)
                self.modalities.append("video")

        if audio_ds is not None:
            for item in audio_ds:
                self.data.append(item)
                self.modalities.append("audio")

        if text_ds is not None:
            for item in text_ds:
                self.data.append(item)
                self.modalities.append("text")
        
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]



class NTTrainer(SFTTrainer):

    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, diy_config: OmniTrainingConfig,processor: Qwen2_5OmniProcessor):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset)
        self.diy_config = diy_config
        self.processor = processor


    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if hasattr(self.train_dataset, 'modalities'):
            modalities = self.train_dataset.modalities
        else:
            modalities = [item['modality'] for item in self.train_dataset]

        sampler = DistributedModalityBatchSampler(
            modalities=modalities,
            batch_size=self.diy_config.per_device_train_batch_size,
            # num_replicas=self.args.world_size,
            # rank=self.args.process_index,
            shuffle=True,
            drop_last=True
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler, 
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self):
        """Get evaluation dataloader"""
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        if hasattr(self.eval_dataset, 'modalities'):
            modalities = self.eval_dataset.modalities
        else:
            modalities = [item['modality'] for item in self.eval_dataset]

        sampler = DistributedModalityBatchSampler(
            modalities=modalities,
            batch_size=self.diy_config.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False
        )

        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(self, eval_dataset: Optional[torch.utils.data.Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        """Evaluate the model on the evaluation dataset"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is None:
            return {}
        
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs in tqdm(eval_dataloader, desc="Evaluating"):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                loss = outputs.loss
                
                # Accumulate loss
                batch_size = inputs["input_ids"].shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_perplexity": perplexity,
        }
        
        self.model.train()
        return metrics


        
class OmniTrainer:
    def __init__(self, config: OmniTrainingConfig):
        self.config = config


    
    def create_dataset(self):
        if self.config.va_data:
            va_ds_train, va_ds_eval = WikiVideoDataset(config=self.config, modality='omni').extract_data()
        else:
            va_ds_train = None
            va_ds_eval = None
        if self.config.video_data:
            video_ds_train, video_ds_eval = WikiVideoDataset(config=self.config, modality='video').extract_data()
        else:
            video_ds_train = None
            video_ds_eval = None
        if self.config.audio_data:
            audio_ds_train, audio_ds_eval = ClothoDataset(config=self.config).extract_data()
        else:
            audio_ds_train = None
            audio_ds_eval = None
        if self.config.text_data:
            text_ds_train, text_ds_eval = UNLI_Dataset(config=self.config).extract_data()
        else:
            text_ds_train = None
            text_ds_eval = None
        self.train_dataset = MultiModalDataset(va_ds_train, video_ds_train, audio_ds_train, text_ds_train)
        self.eval_dataset = MultiModalDataset(va_ds_eval, video_ds_eval, audio_ds_eval, text_ds_eval)


    def create_model(self):
        # self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(self.config.model_name, torch_dtype="auto",device_map="auto")
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(self.config.model_name, torch_dtype="auto")

        # self.model.disable_talker()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_name)
        self.freeze_other_modules()


    def setup_lora(self):
        model_head = ["lm_head"]
        all_modules = ["q_proj", "k_proj", "v_proj", "o_proj","out_proj","q","k","v","o","lm_head"]
        video_modules = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name.startswith("visual.blocks"):
                if ".attn.q" in name or ".attn.k" in name or ".attn.v" in name or ".attn.proj" in name:
                    video_modules.append(name)

        audio_modules = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name.startswith("audio_tower.layers"):
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    audio_modules.append(name)

        text_modules = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name.startswith("model.layers"):
                if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    text_modules.append(name)
        
        self.lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj",'q','k','v','o','lm_head'],
            target_modules=text_modules + ["lm_head"],
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def create_trainer(self):
        self.sft_config = self.create_training_args()
        if self.config.use_lora:
            self.setup_lora()
        self.model.config.vocab_size = self.model.config.text_config.vocab_size
        # import pdb; pdb.set_trace()
        self.trainer = NTTrainer(
            model=self.model,
            args=self.sft_config,
            data_collator=CollateFn(self.processor),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            diy_config=self.config,
            processor=self.processor,
            # compute_metrics=compute_metrics,
        )
    
    def create_training_args(self):
        return SFTConfig(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            # evaluation_strategy=self.config.evaluation_strategy if self.config.do_eval else "no",
            # eval_steps=self.config.eval_steps if self.config.do_eval and self.config.evaluation_strategy == "steps" else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=self.config.remove_unused_columns,
            push_to_hub=self.config.push_to_hub,
            report_to=self.config.report_to,
            warmup_steps=self.config.warmup_steps,
            # load_best_model_at_end=self.config.load_best_model_at_end if self.config.do_eval else False,
            # metric_for_best_model=self.config.metric_for_best_model if self.config.do_eval else None,
            deepspeed='./ds_config.json'
        )

    def train(self):
        self.trainer.train()

    def freeze_audio_encoder(self):
        for name, param in self.model.named_parameters():
            if 'audio_tower' in name:
                param.requires_grad = False
    
    
    def freeze_vision_encoder(self):
        for name, param in self.model.named_parameters():
            if 'visual' in name:
                param.requires_grad = False

    def freeze_other_modules(self):
        for name, param in self.model.named_parameters():
            if 'talker.' in name or 'token2wav.' in name:
                param.requires_grad = False




if __name__ == "__main__":
    config = OmniTrainingConfig()
    trainer = OmniTrainer(config)
    trainer.create_dataset()
    trainer.create_model()
    # trainer.setup_lora()
    trainer.create_trainer()
    trainer.train()
