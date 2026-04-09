from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
from config import OmniTrainingConfig
from typing import List, Dict, Any, Optional
import torch.nn as nn
import torch
from torch.utils.data import Sampler
import numpy as np
# from dataset import WikiVideoDataset
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from mmdataset import WikiVideoDataset, ClothoDataset, UNLI_Dataset, FakeDataset
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

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
        map_ ={
            "omni": 0,
            "video": 1,
            "audio": 2,
            "text": 3
        }
        texts = [message['messages'] for message in batch]
        modality = [message['modality'] for message in batch][0]

        if modality == "omni":
            try:
                USE_AUDIO_IN_VIDEO = True
                text = self.processor.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)


            except Exception as e:
                USE_AUDIO_IN_VIDEO = False
                text = self.processor.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
                audio, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
                inputs = self.processor(text=text, audio=None, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        elif modality == "video":
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
            audio, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=None, images=None, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        elif modality == "audio":
            USE_AUDIO_IN_VIDEO = True
            text = self.processor.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
            audio, images, videos = process_mm_info(texts, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(text=text, audio=audio, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        elif modality == "text":
            USE_AUDIO_IN_VIDEO = False
            text = self.processor.apply_chat_template(texts, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, audio=None, images=None, videos=None, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        inputs['use_audio_in_video'] = USE_AUDIO_IN_VIDEO

        inputs['answer'] = torch.tensor([message['answer'] for message in batch])
        inputs['modality'] = torch.tensor([map_[message['modality']] for message in batch])
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

def get_param_sum(model):
    total = 0.0
    for param in model.parameters():
        total += param.data.sum().item()
    return total

def generate_soft_label(target_prob, num_bins=100, sigma=0.05, device='cpu'):
    bins = torch.arange(num_bins, device=device).float()
    
    target_idx = target_prob * (num_bins - 1)
    
    density = torch.exp(- (bins - target_idx)**2 / (2 * sigma**2))

    soft_label = density / density.sum()
    
    return soft_label

class KLTrainer(SFTTrainer):

    def __init__(self, model, args, data_collator, train_dataset, eval_dataset, diy_config: OmniTrainingConfig,processor: Qwen2_5OmniProcessor):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset)
        self.diy_config = diy_config
        self.processor = processor
        self.param_sum = get_param_sum(model)

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


    def compute_loss(self, model, inputs,num_items_in_batch):
        modality = inputs.pop("modality")

        inputs = inputs.to(model.device)
        output = model(**inputs)
        logits = output.logits
        # batch_size = input_ids.shape[0]
        # for i in range(min(2, batch_size)):
        #     last_5_ids = input_ids[i, -100:].tolist()
        #     decoded_last_5 = self.processor.tokenizer.decode(last_5_ids)
        #     print(f"Sample {i} Last 5 tokens: {last_5_ids} | Decoded: '{decoded_last_5}'")
        CONF_tokens = [self.diy_config.new_token_prefix.format(idx=idx) for idx in range(self.diy_config.new_token_num)]
        CONF_token_ids = self.processor.tokenizer.convert_tokens_to_ids(CONF_tokens)
        target_floats = inputs.pop("answer")
        logits = output.logits[:, -1, :]

        relevant_logits = logits[:, CONF_token_ids]
        logprob = F.log_softmax(relevant_logits, dim=-1)
        batch_soft_labels = []
        for val in target_floats:
            batch_soft_labels.append(generate_soft_label(val, num_bins=self.diy_config.new_token_num, sigma=self.diy_config.sigma, device=model.device))
        bins = self.diy_config.new_token_num
        # step_size = 1.0 / bins

        target_dist = torch.stack(batch_soft_labels)



        probabilities = torch.exp(logprob)
        # expectation_values = torch.tensor([i * step_size + 0.5 * step_size for i in range(bins)],device=logits.device)
        scale_values = torch.linspace(0.0, 1.0, self.diy_config.new_token_num).to(model.device)
        target_scores = torch.tensor(target_floats, dtype=torch.float32).to(model.device)
        pred_scores = (probabilities * scale_values).sum(dim=-1)

        pre_pdf = torch.cumsum(probabilities, dim=-1)
        target_pdf = torch.cumsum(target_dist, dim=-1)
        # emd_loss = torch.mean((pre_pdf - target_pdf)**2)
        loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss = loss_fct(logprob, target_dist)
        loss = kl_loss


        return loss

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
        all_pred_scores = []
        all_target_scores = []
        
        CONF_tokens = [self.diy_config.new_token_prefix.format(idx=idx) for idx in range(self.diy_config.new_token_num)]
        CONF_token_ids = self.processor.tokenizer.convert_tokens_to_ids(CONF_tokens)
        scale_values = torch.linspace(0.0, 1.0, self.diy_config.new_token_num).to(self.model.device)
        
        with torch.no_grad():
            for inputs in tqdm(eval_dataloader, desc="Evaluating"):
                modality = inputs.pop("modality")
                target_floats = inputs.pop("answer")
                inputs = inputs.to(self.model.device)
                target_floats = target_floats.to(self.model.device)
                
                output = self.model(**inputs)
                logits = output.logits[:, -1, :]
                
                relevant_logits = logits[:, CONF_token_ids]
                logprob = F.log_softmax(relevant_logits, dim=-1)
                
                # Generate soft labels for loss calculation
                batch_soft_labels = []
                for val in target_floats:
                    batch_soft_labels.append(generate_soft_label(val, num_bins=self.diy_config.new_token_num, sigma=self.diy_config.sigma, device=self.model.device))
                target_dist = torch.stack(batch_soft_labels)
                
                # Calculate loss
                loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
                kl_loss = loss_fct(logprob, target_dist)
                
                # Calculate predictions
                probabilities = torch.exp(logprob)
                pred_scores = (probabilities * scale_values).sum(dim=-1)
                
                total_loss += kl_loss.item() * len(target_floats)
                total_samples += len(target_floats)
                
                all_pred_scores.extend(pred_scores.cpu().numpy())
                all_target_scores.extend(target_floats.cpu().numpy())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        all_pred_scores = np.array(all_pred_scores)
        all_target_scores = np.array(all_target_scores)
        
        # Calculate metrics
        mae = np.mean(np.abs(all_pred_scores - all_target_scores))
        mse = np.mean((all_pred_scores - all_target_scores) ** 2)
        rmse = np.sqrt(mse)
        
        # Binary accuracy (threshold at 0.5)
        pred_binary = (all_pred_scores > 0.5).astype(int)
        target_binary = (all_target_scores > 0.5).astype(int)
        accuracy = np.mean(pred_binary == target_binary)
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_mae": mae,
            f"{metric_key_prefix}_mse": mse,
            f"{metric_key_prefix}_rmse": rmse,
            f"{metric_key_prefix}_accuracy": accuracy,
        }
        
        self.model.train()
        return metrics

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        # This method is called by Trainer if compute_metrics is provided
        # For now, we handle metrics in evaluate() method
        return {}
    
def create_fake_dataset(modality: str):
    if modality == "omni":
        return FakeDataset(modality="omni").extract_data()
    elif modality == "video":
        return FakeDataset(modality="video").extract_data()
    elif modality == "audio":
        return FakeDataset(modality="audio").extract_data()
    elif modality == "text":
        return FakeDataset(modality="text").extract_data()
    


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


    def create_model(self):
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(self.config.model_name, torch_dtype="auto")
        # self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(self.config.model_name, torch_dtype="auto",device_map="auto")
        
        # self.model.config.vocab_size = self.model.config.text_config.vocab_size
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.model_name)


        # add new tokens for the model
        new_tokens = [self.config.new_token_prefix.format(idx=idx) for idx in range(self.config.new_token_num)]
        self.processor.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        output_embeddings = self.model.get_output_embeddings().weight

        new_token_ids = self.processor.tokenizer.convert_tokens_to_ids(new_tokens)
        with torch.no_grad():
            for i in range(len(new_token_ids)):
                
                tenth = i//10
                digit = i%10
                tenth_id = self.processor(str(tenth)).input_ids[0][0]
                digit_id = self.processor(str(digit)).input_ids[0][0]
                tenth_embedding = output_embeddings[tenth_id]
                digit_embedding = output_embeddings[digit_id]
                output_embeddings[new_token_ids[i]] = 10/11*tenth_embedding + 1/11*digit_embedding
        
        self.processor.save_pretrained(self.config.output_dir)


    def create_trainer(self):
        self.sft_config = self.create_training_args()
        if self.config.use_lora:
            self.setup_lora()

        self.model.config.vocab_size = self.model.config.text_config.vocab_size

        self.trainer = KLTrainer(
            model=self.model,
            args=self.sft_config,
            data_collator=CollateFn(self.processor),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            diy_config=self.config,
            processor=self.processor
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
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=self.config.remove_unused_columns,
            push_to_hub=self.config.push_to_hub,
            report_to=self.config.report_to,
            warmup_steps=self.config.warmup_steps,
            # evaluation_strategy=self.config.evaluation_strategy,
            # eval_steps=self.config.eval_steps if self.config.evaluation_strategy == "steps" else None,
            # load_best_model_at_end=self.config.load_best_model_at_end,
            # metric_for_best_model=self.config.metric_for_best_model,
            deepspeed='ds_config.json'
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




if __name__ == "__main__":

    config = OmniTrainingConfig()
    trainer = OmniTrainer(config)
    trainer.create_dataset()

    trainer.create_model()
    trainer.create_trainer()
    trainer.train()
