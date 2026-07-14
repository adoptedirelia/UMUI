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
│   ├── run_eval.py                                   # Universal eval: released UNLI model on UMUI benchmarks
│   └── main.py                                       # Metric utilities for existing prediction files
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
export CUDA_HOME=/usr/local/cuda-12.8
deepspeed --num_gpus=4 omni_trainer.py
```

### NT Training

1. Edit `src/training/NT/config.py` — set data paths
2. Run:

```bash
cd src/training/NT
export CUDA_HOME=/usr/local/cuda-12.8
deepspeed --num_gpus=4 omni_trainer.py
```

## Evaluation

`src/eval/run_eval.py` evaluates the released model ([AdoptedIrelia/UNLI](https://huggingface.co/AdoptedIrelia/UNLI)) on the UMUI benchmarks ([AdoptedIrelia/UMUI](https://huggingface.co/datasets/AdoptedIrelia/UMUI)) in one command. The model (base + LoRA adapter in `lora/`) and the label files are downloaded from Hugging Face automatically; only the media tarballs (`wikivideo.tar.gz`, `audio_entailment.tar.gz`) need to be extracted locally and passed via `--media-root` (or the `UMUI_MEDIA_ROOT` env var).

### Quick start via scripts

```bash
# one-time: download model + labels + media tarballs from Hugging Face and extract
bash script/download.sh /path/to/UMUI-data        # SKIP_MEDIA=1 to skip the ~26 GB tarballs

# run the evals (extra args pass through to run_eval.py)
export UMUI_ROOT=/path/to/UMUI-data
bash script/eval_unli.sh
bash script/eval_clotho.sh
bash script/eval_wikivideo.sh video     # or: audio / omni
```

### Running run_eval.py directly

```bash
# text: UNLI validation split (no media needed)
python -m src.eval.run_eval --dataset unli --modality text

# audio: Clotho entailment
python -m src.eval.run_eval --dataset clotho --modality audio \
    --media-root /path/to/extracted/media

# video / audio / omni: WikiVideo (media-root = the extracted wikivideo.tar.gz)
python -m src.eval.run_eval --dataset wikivideo --modality video \
    --media-root /path/to/wikivideo_
```

WikiVideo items are built the same way as `src/training/NLI/evaldataset.py` — from `<media-root>/annotations/final_data_2015-2025.json`, restricted to the 10 eval events, with per-modality labels (`video|ocr` for video, `audio & !video & !ocr` for audio, `video|audio` for omni). Human probabilities are joined in from `UMUI-annotation/annotation.json` (downloaded from the hub, or `--annotation-file`) for the soft metrics.

Useful options:

| Option | Description |
|--------|-------------|
| `--model` | HF repo or local dir; a `lora/` subfolder (or `model/` + `lora/` layout) is auto-detected. Default: `AdoptedIrelia/UNLI` |
| `--lora` | Explicit LoRA adapter path, overrides auto-detection |
| `--data-file` | Use a local label file instead of the default source |
| `--annotation-file` | WikiVideo only: local `annotation.json` with human probabilities |
| `--batch-size` | Inference batch size (default 4) |
| `--start` / `--end` | Dataset slice, for sharding across SLURM jobs |
| `--metrics-only` | Recompute metrics from an existing output file, no GPU needed |

Predictions are appended to `<output-dir>/<dataset>_<modality>.jsonl` as they are produced, so an interrupted run resumes where it left off; metrics land next to it in `<dataset>_<modality>.metrics.json`.

Reported metrics: accuracy / F1 / precision / recall on all datasets; NLL, ECE, MSE, and Krippendorff's α are additionally computed against the human probability on the annotated subset (all of UNLI; for WikiVideo the items matched in `annotation.json`, reported as `n_with_probability`). Krippendorff's α requires the optional `krippendorff` package.

Legacy metric utilities for existing prediction files:

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
