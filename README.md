# UMUI

Multi-modal claim verification using distilled Natural Language Inference. Supports synthetic data generation, model fine-tuning, and evaluation across video, audio, text, and omni modalities.

## Project Structure

```
src/
├── prompt.py / prompt_binary.py / prompt_score.py   # Prompt templates (continuous / binary / score)
├── run_synthetic.py                                  # CLI entry for synthetic data generation
├── analyse/                                          # Analysis and metrics
│   ├── analyse.py                                    # Core utilities (correlation, Krippendorff α, histograms)
│   └── main.py                                       # Analysis pipeline (variant 1)
├── eval/
│   └── main.py                                       # Evaluation (acc, F1, MSE, NLL, ECE)
├── synthetic_data/                                   # Synthetic data generation
│   ├── config.py                                     # Configuration & CLI arguments
│   ├── generation.py                                 # Core batch generation pipeline
│   ├── mmdataset.py                                  # Dataset loaders (WikiVideo, Clotho, UNLI, etc.)
│   ├── engine/                                       # Inference engines per model family
│   └── score_engine/                                 # Score-based (0-9 token) inference engines
└── training/                                         # Model fine-tuning (Qwen2.5-Omni)
    ├── NLI/                                          # Single probability-token training
    └── NT/                                           # Natural-language probability training
```

## Environment Setup

### Qwen3-VL

```bash
conda create -n qwen3vl python=3.10
conda activate qwen3vl
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.11.0
pip install qwen-vl-utils[decord]
pip install datasets
pip install -U openai-whisper
pip install transformers==4.57.6
```

### Qwen3-Omni

```bash
conda create -n qwen3omni python=3.12
conda activate qwen3omni
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install vllm==0.13.0
pip install qwen-omni-utils[decord]
pip install datasets
```

## Data Generation

### Basic Usage

```bash
python -m src.run_synthetic \
    --model <model_name> \
    --modality <video|audio|text|omni> \
    --dataset_name <wikivideo|clotho|unli|peopleprofile|violin> \
    --output_path <output_path> \
    --batch_size <batch_size> \
    --wikivideo_pre_path <path_to_data> \
    --wikivideo_label_path <path_to_labels>
```

### Examples

**Video (continuous probability):**

```bash
python -m src.run_synthetic \
    --model "Qwen/Qwen3-VL-32B-Thinking" \
    --dataset_name wikivideo \
    --modality video \
    --output_path "./result/qwen3vl_32b.json" \
    --batch_size 4 \
    --response_num 10
```

**Omni (video + audio):**

```bash
python -m src.run_synthetic \
    --model "Qwen/Qwen3-Omni-30B-A3B-Instruct" \
    --dataset_name wikivideo \
    --modality omni \
    --output_path "./result/qwen3_omni_30b.json" \
    --batch_size 2
```

**Binary evaluation:**

```bash
python -m src.run_synthetic \
    --modality video \
    --model "Qwen/Qwen3-VL-32B-Instruct" \
    --dataset_name wikivideo \
    --output_path "./eval_result/qwen3_vl_32b.json" \
    --batch_size 4 \
    --evaluate True \
    --binary True
```

### SLURM Array Jobs

Split work across array tasks with `--array_job_id` and `--array_total_jobs`:

```bash
#!/bin/bash
#SBATCH --array=0-1
#SBATCH --gres=gpu:a100:2

python -m src.run_synthetic \
    --model "Qwen/Qwen3-VL-32B-Thinking" \
    --dataset_name wikivideo \
    --modality video \
    --output_path "./result/output_${SLURM_ARRAY_TASK_ID}.json" \
    --batch_size 2 \
    --tensor_parallel_size 2 \
    --array_job_id $SLURM_ARRAY_TASK_ID \
    --array_total_jobs 2
```

## Training

Two training paradigms are supported:

| Method | Description | Output Format |
|--------|-------------|---------------|
| **NLI** | Single probability token via learned `<CON_*>` tokens | Token distribution → scalar |
| **NT**  | Natural language response with `<answer>0.x</answer>` | Free-form text |

### NLI Training

1. Edit `src/training/NLI/config.py` — set data paths and modality flags (`va_data`, `video_data`, `audio_data`, `text_data`)
2. Run:

```bash
cd src/training/NLI
deepspeed --num_gpus=4 omni_trainer.py
```

### NT Training

1. Edit `src/training/NT/config.py` — set data paths
2. Run:

```bash
cd src/training/NT
deepspeed --num_gpus=4 omni_trainer.py
```

## Evaluation

```bash
python -m src.eval.main
```

### Analysis

Generates correlation maps, distribution histograms, and Krippendorff's α:

```bash
python -m src.analyse.main
```

## Data Format

Each data item follows this schema:

```json
{
    "path": "video/audio file path",
    "label": true,
    "claim": "The event occurred on ...",
    "type": "event_category",
    "modality": "video"
}
```
