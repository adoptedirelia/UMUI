import os
import argparse
from dataclasses import dataclass
from typing import List, Optional
from dataclasses import dataclass, field

DATA_ROOT = os.environ.get("DISTILL_UNLI_DATA_ROOT", "")

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class AppConfig:
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    cache_dir: str = None
    output_path: str = "./result/synthetic_data_with_prompt.json"
    backend: str = "vllm"  # choices: "vllm" | "hf"
    max_new_tokens: int = 4096
    evaluate: bool = False
    binary: bool = False
    data_split: int = -1
    array_job_id: int = 0
    array_total_jobs: int = -1
    # vLLM specific
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.7
    limit_mm_video_per_prompt: int = 1
    batch_size: int = 4
    max_pixels: int = 256 * 256
    min_pixels: int = 256 * 256
    fps: float = 0.5
    modality: str = 'video'
    generate_score: bool = False
    human_data_path: str = ''
    human_only: bool = False
    response_num: int = 1
    temperature: float = 2
    top_p: float = 0.95
    dataset_name: str = 'wikivideo'
    # wikivideo 
    wikivideo_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/combined_videos")
    wikivideo_audio_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/audios/en")
    wikivideo_video_pre_path: str = os.path.join(DATA_ROOT, "wikivideo/videos/en")

    wikivideo_label_path: str = "./data_transfer/final_data_2015-2025.json"
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

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./result/synthetic_data_with_prompt.json")
    parser.add_argument("--binary", type=bool, default=False)
    parser.add_argument("--backend", type=str, choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--limit_mm_video_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=128*128)
    parser.add_argument("--min_pixels", type=int, default=32*32)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--modality", type=str, choices=["video", "text", "audio","omni"], default="video")
    parser.add_argument("--generate_score", type=bool, default=False)
    parser.add_argument("--human_data_path", type=str, default='')
    parser.add_argument("--human_only", type=bool, default=False)
    parser.add_argument("--response_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dataset_name", type=str, choices=["wikivideo", "clotho", "unli", "peopleprofile", "violin"], default="wikivideo")
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--wikivideo_pre_path", type=str, default=os.path.join(DATA_ROOT, "wikivideo/combined_videos"))
    parser.add_argument("--wikivideo_label_path", type=str, default='./data_transfer/final_data_2015-2025.json')
    parser.add_argument("--wikivideo_audio_pre_path", type=str, default=os.path.join(DATA_ROOT, "wikivideo/audios/en"))
    parser.add_argument("--wikivideo_video_pre_path", type=str, default=os.path.join(DATA_ROOT, "wikivideo/videos/en"))
    parser.add_argument("--clotho_pre_path", type=str, default=os.path.join(DATA_ROOT, "audio_entailment/clotho"))
    parser.add_argument("--clotho_label_path", type=str, default=os.path.join(DATA_ROOT, "AudioEntailment/data/CLE"))
    parser.add_argument("--unli_label_path", type=str, default=os.path.join(DATA_ROOT, "UNLI"))
    parser.add_argument("--peopleprofile_label_path", type=str, default=os.path.join(DATA_ROOT, "peopleprofile"))
    parser.add_argument("--violin_label_path", type=str, default=os.path.join(DATA_ROOT, "violin/violin_annotation.json"))
    parser.add_argument("--violin_pre_path", type=str, default=os.path.join(DATA_ROOT, "violin/violin_videos"))
    parser.add_argument("--data_split", type=int, default=-1)
    parser.add_argument("--array_job_id", type=int, default=0)
    parser.add_argument("--array_total_jobs", type=int, default=-1)
    return parser


def parse_config() -> AppConfig:
    args = build_arg_parser().parse_args()
    print(args)
    return AppConfig(
        model=args.model,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        binary=args.binary,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_video_per_prompt=args.limit_mm_video_per_prompt,
        batch_size=args.batch_size,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        fps=args.fps,
        modality=args.modality,
        generate_score=args.generate_score,
        human_data_path=args.human_data_path,
        human_only=args.human_only,
        response_num=args.response_num,
        temperature=args.temperature,
        top_p=args.top_p,
        dataset_name=args.dataset_name,
        evaluate=args.evaluate,
        wikivideo_pre_path=args.wikivideo_pre_path,
        wikivideo_label_path=args.wikivideo_label_path,
        wikivideo_audio_pre_path=args.wikivideo_audio_pre_path,
        wikivideo_video_pre_path=args.wikivideo_video_pre_path,
        clotho_pre_path=args.clotho_pre_path,
        clotho_label_path=args.clotho_label_path,
        unli_label_path=args.unli_label_path,
        peopleprofile_label_path=args.peopleprofile_label_path,
        violin_label_path=args.violin_label_path,
        violin_pre_path=args.violin_pre_path,
        data_split=args.data_split,
        array_job_id=args.array_job_id,
        array_total_jobs=args.array_total_jobs,
    )
