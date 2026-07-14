import os
from dataclasses import dataclass
from typing import Optional
from dataclasses import field

DATA_ROOT = os.environ.get("DISTILL_UNLI_DATA_ROOT", "")

@dataclass
class OmniTrainingConfig:
    model_name: str = 'Qwen/Qwen2.5-Omni-3B'
    train_data_path: str = ''
    output_dir: str = './output/NT_output_lora'

    # train_data
    omni_path: str = ''
    video_path: str = ''
    audio_path: str = ''
    va_data: bool = False
    video_data: bool = True
    audio_data: bool = True 
    text_data: bool = True 
    # Training parameters
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 15
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 2
    do_eval: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    save_steps: int = 100
    save_total_limit: int = 2
    remove_unused_columns: bool = False
    push_to_hub: bool = False
    report_to: str = "wandb"
    wandb_project: str = "NT"
    wandb_run: str = "NT"
    completion_only_loss: bool = False
    warmup_steps: int = 2000
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[str] = None

    # dataset parameters
    # wikivideo 
    wikivideo_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/combined_videos")
    wikivideo_audio_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/audios/en")
    wikivideo_label_path: str = os.path.join(DATA_ROOT, "wikivideo/annotations/final_data_2015-2025.json")
    wikivideo_eval_tag: list = field(default_factory=lambda:
        [
            'Launch and commissioning of the James Webb Space Telescope',
            '2018 lower Puna eruption',
            'Notre-Dame fire',
            '2022 United States Senate election in Georgia',
            'Hurricane Irma',
            '2018 Anchorage earthquake',
            '2025_Canadian_federal_election',
            '2025_Myanmar_earthquake',
            'Blue_Ghost_Mission_1',
            'Liberation_Day_Tariffs'
        ])
    wikivideo_ratio: float = 1

    # clotho
    clotho_pre_path: str = os.path.join(DATA_ROOT, "audio_entailment/clotho")
    clotho_label_path: str = os.path.join(DATA_ROOT, "AudioEntailment/data/CLE")
    clotho_eval_tag: list = field(default_factory=lambda: ['evaluation'])
    # UNLI
    unli_label_path: str = os.path.join(DATA_ROOT, "UNLI")
    unli_eval_tag: list = field(default_factory=lambda: ['validation'])
    # PeopleProfile
    peopleprofile_label_path: str = os.path.join(DATA_ROOT, "peopleprofile")
    peopleprofile_eval_tag: list = field(default_factory=lambda: ['dev'])
    # VIOLIN 
    violin_label_path: str = os.path.join(DATA_ROOT, "violin/violin_annotation.json")
    violin_pre_path: str = os.path.join(DATA_ROOT, "violin/violin_videos")
    violin_eval_tag: list = field(default_factory=lambda: ['validate'])

@dataclass
class OmniEvalConfig:
    model_name: str = 'Qwen/Qwen2.5-Omni-3B'
    video_path: str = os.path.join(DATA_ROOT, "wikivideo/combined_videos")
    output_eval_path: str = './output/NT_eval'
    lora_path: str = ''
    processor_path: str = 'Qwen/Qwen2.5-Omni-3B'
    batch_size: int = 4
    # New tokens for the model
    new_token_num: int = 100
    new_token_prefix: str = "<CON_{idx}>"
    sigma: float = 0.05

    dataset_name: str = 'wikivideo'
    modality: str = 'text'
    # wikivideo 
    wikivideo_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/combined_videos")
    wikivideo_audio_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/audios/en")
    wikivideo_label_path: str = os.path.join(DATA_ROOT, "wikivideo/annotations/final_data_2015-2025.json")
    wikivideo_eval_tag: list = field(default_factory=lambda:
        [
            'Launch and commissioning of the James Webb Space Telescope',
            '2018 lower Puna eruption',
            'Notre-Dame fire',
            '2022 United States Senate election in Georgia',
            'Hurricane Irma',
            '2018 Anchorage earthquake',
            '2025_Canadian_federal_election',
            '2025_Myanmar_earthquake',
            'Blue_Ghost_Mission_1',
            'Liberation_Day_Tariffs'
        ])
    # clotho
    clotho_pre_path: str = os.path.join(DATA_ROOT, "audio_entailment/clotho")
    clotho_label_path: str = os.path.join(DATA_ROOT, "AudioEntailment/data/CLE")
    clotho_eval_tag: list = field(default_factory=lambda: ['evaluation'])
    # UNLI
    unli_label_path: str = os.path.join(DATA_ROOT, "UNLI")
    unli_eval_tag: list = field(default_factory=lambda: ['validation'])
    # PeopleProfile
    peopleprofile_label_path: str = os.path.join(DATA_ROOT, "peopleprofile")
    peopleprofile_eval_tag: list = field(default_factory=lambda: ['dev'])
    # VIOLIN 
    violin_label_path: str = os.path.join(DATA_ROOT, "violin/violin_annotation.json")
    violin_pre_path: str = os.path.join(DATA_ROOT, "violin/violin_videos")
    violin_eval_tag: list = field(default_factory=lambda: ['validate'])
