"""
Universal evaluation for the UNLI model on the UMUI benchmarks.

Model:   https://huggingface.co/AdoptedIrelia/UNLI
         (Qwen2.5-Omni-3B thinker with <CON_i> probability tokens at the repo
         root, LoRA adapter in the `lora/` subfolder)
Dataset: https://huggingface.co/datasets/AdoptedIrelia/UMUI
         (label files are downloaded automatically from the hub; the large
         media tarballs — wikivideo.tar.gz / audio_entailment.tar.gz — must be
         extracted locally and passed via --media-root)

Examples
--------
# text: UNLI validation split (no media needed)
python -m src.eval.run_eval --dataset unli --modality text

# audio: Clotho entailment
python -m src.eval.run_eval --dataset clotho --modality audio \
    --media-root /exp/dzhang1/UMUI/training-data

# video / audio / omni: WikiVideo
python -m src.eval.run_eval --dataset wikivideo --modality video \
    --media-root /exp/dzhang1/UMUI/training-data

# recompute metrics from a finished run, no GPU needed
python -m src.eval.run_eval --dataset wikivideo --modality video --metrics-only

Predictions are appended to <output-dir>/<dataset>_<modality>.jsonl as they are
produced, so an interrupted run resumes where it left off. Metrics are written
to <output-dir>/<dataset>_<modality>.metrics.json.
"""

import argparse
import json
import os
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "AdoptedIrelia/UNLI"
DEFAULT_DATA_REPO = "AdoptedIrelia/UMUI"
THRESHOLD = 0.5
MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wav", ".mp3", ".flac", ".m4a"}

# ---------------------------------------------------------------------------
# Prompts (identical to src/training/NLI/prompt.py — the model was trained on
# these exact strings, do not edit casually)
# ---------------------------------------------------------------------------

NLI_SYSTEM_PROMPT = """
You are a probability estimation model.
Analyze the video and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT = """
Claim: {text}
"""

NLI_SYSTEM_PROMPT_AUDIO = """
You are a probability estimation model.
Analyze the audio and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT_AUDIO = """
Claim: {text}
"""

NLI_SYSTEM_PROMPT_TEXT = """
You are a probability estimation model.
Analyze the sentence and the claim.
Output ONLY one single probability token representing the likelihood that the claim is true.
Do not output any other text or explanation.
"""

NLI_PROMPT_TEXT = """
Sentence: {sentence}
Claim: {claim}
"""


def build_messages(item: dict, modality: str) -> list:
    if modality == "text":
        system, user_text = NLI_SYSTEM_PROMPT_TEXT, NLI_PROMPT_TEXT.format(
            sentence=item["sentence"], claim=item["claim"])
        user_content = [{"type": "text", "text": user_text}]
    elif modality == "audio":
        system = NLI_SYSTEM_PROMPT_AUDIO
        user_content = [
            {"type": "text", "text": NLI_PROMPT_AUDIO.format(text=item["claim"])},
            {"type": "audio", "audio": item["path"]},
        ]
    else:  # video / omni share the video prompt; omni additionally feeds the audio track
        system = NLI_SYSTEM_PROMPT
        user_content = [
            {"type": "text", "text": NLI_PROMPT.format(text=item["claim"])},
            {"type": "video", "video": item["path"],
             "resized_height": 256, "resized_width": 256, "fps": 0.5},
        ]
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def hub_file(repo: str, filename: str, override: str = None) -> str:
    """Return a local path for a label file: --data-file override or hub download."""
    if override:
        return override
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


class MediaIndex:
    """Resolves the media paths stored in label files against a local media root."""

    def __init__(self, root: str):
        self.root = root
        self.by_name = {}
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in MEDIA_EXTS:
                    self.by_name.setdefault(name, os.path.join(dirpath, name))
        if not self.by_name:
            raise FileNotFoundError(f"no media files found under --media-root {root}")

    def resolve(self, path: str) -> str:
        direct = os.path.join(self.root, path)
        if os.path.exists(direct):
            return direct
        return self.by_name.get(os.path.basename(path))


def load_unli(repo: str, split: str, data_file: str) -> list:
    path = hub_file(repo, f"training-data/unli/{split}.jsonl", data_file)
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items.append({
                "sentence": d["premise"],
                "claim": d["hypothesis"],
                "label": d["label"] > THRESHOLD,
                "probability": float(d["label"]),
                "type": split,
                "modality": "text",
            })
    return items


def load_clotho(repo: str, data_file: str) -> list:
    # The claim/label/path lists live inside the released prediction files; we
    # use one as the item source and ignore its `answer` field.
    path = hub_file(repo, "UMUI-binary/clotho/CLUE-D.json", data_file)
    with open(path) as f:
        data = json.load(f)
    return [{
        "claim": d["claim"],
        "label": bool(d["label"]),
        "path": d["path"],
        "type": d["type"],
        "modality": "audio",
    } for group in data.values() for d in group]


# the eval split, same as wikivideo_eval_tag in src/training/NLI/config.py
WIKIVIDEO_EVAL_EVENTS = [
    "Launch and commissioning of the James Webb Space Telescope",
    "2018 lower Puna eruption",
    "Notre-Dame fire",
    "2022 United States Senate election in Georgia",
    "Hurricane Irma",
    "2018 Anchorage earthquake",
    "2025_Canadian_federal_election",
    "2025_Myanmar_earthquake",
    "Blue_Ghost_Mission_1",
    "Liberation_Day_Tariffs",
]


def load_wikivideo(args) -> list:
    """Item construction mirrors src/training/NLI/evaldataset.py::WikiVideoDataset;
    the human probability is joined in from UMUI-annotation/annotation.json
    (the reference file of the original eval_mse_wikivideo)."""
    modality = args.modality
    path = args.data_file
    if not path:
        if not args.media_root:
            raise SystemExit("wikivideo needs --media-root (or --data-file) to locate annotations/final_data_2015-2025.json")
        path = os.path.join(args.media_root, "annotations", "final_data_2015-2025.json")
    with open(path) as f:
        data = json.load(f)

    items = []
    for event in WIKIVIDEO_EVAL_EVENTS:
        samples = data[event]
        seen = set()
        for sentence, claims in zip(samples["original_article"], samples["claims"]):
            for claim in claims:
                modalities = samples["claims_to_supporting_videos"][claim]["videos_modalities"]
                for video, lab in modalities.items():
                    # relative paths match wikivideo_pre_path / wikivideo_audio_pre_path
                    # in the training config, so direct resolution wins over the
                    # basename fallback (data/videos/en holds audio-less duplicates)
                    if modality == "video":
                        label = lab["video"] or lab["ocr"]
                        media = f"data/combined_videos/{video}.mp4"
                    elif modality == "audio":
                        label = lab["audio"] and not lab["video"] and not lab["ocr"]
                        media = f"data/audios/en/{video}.wav"
                    else:  # omni
                        label = lab["video"] or lab["audio"]
                        media = f"data/combined_videos/{video}.mp4"
                    if (media, claim) in seen:
                        continue
                    seen.add((media, claim))
                    items.append({
                        "sentence": sentence,
                        "claim": claim,
                        "label": bool(label),
                        "path": media,
                        "type": event,
                        "modality": modality,
                    })

    try:
        ann_path = hub_file(args.data_repo, "UMUI-annotation/annotation.json", args.annotation_file)
        with open(ann_path) as f:
            probability = {}
            for a in json.load(f):  # duplicates: keep the first, like the original join
                probability.setdefault((a["claim"], os.path.splitext(a["path"])[0]), float(a["probability"]))
        for item in items:
            p = probability.get((item["claim"], os.path.splitext(os.path.basename(item["path"]))[0]))
            if p is not None:
                item["probability"] = p
    except Exception as e:
        print(f"warning: could not join human probabilities ({e}); soft metrics will be skipped")
    return items


def load_items(args) -> list:
    if args.dataset == "unli":
        items = load_unli(args.data_repo, args.split, args.data_file)
    elif args.dataset == "clotho":
        items = load_clotho(args.data_repo, args.data_file)
    elif args.dataset == "wikivideo":
        items = load_wikivideo(args)
    else:
        raise ValueError(f"unknown dataset {args.dataset}")

    seen, deduped = set(), []
    for item in items:
        key = (item.get("path", ""), item.get("sentence", ""), item["claim"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    if args.modality != "text":
        if not args.media_root:
            raise SystemExit(f"--media-root is required for {args.dataset}/{args.modality}")
        index = MediaIndex(args.media_root)
        resolved, missing = [], 0
        for item in deduped:
            local = index.resolve(item["path"])
            if local is None:
                missing += 1
                continue
            resolved.append({**item, "path": local})
        if missing:
            print(f"warning: {missing}/{len(deduped)} items skipped (media not found under {args.media_root})")
        deduped = resolved

    end = args.end if args.end >= 0 else len(deduped)
    return deduped[args.start:end]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UNLIScorer:
    def __init__(self, model_path: str, lora_path: str, device: str, attn: str):
        import torch
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        from peft import PeftModel
        from qwen_omni_utils import process_mm_info

        self.torch = torch
        self.process_mm_info = process_mm_info

        # accept either a repo with base weights at the root (HF layout) or a
        # local dir holding `model/` + `lora/` subfolders
        base_path = model_path
        if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")) \
                and os.path.exists(os.path.join(model_path, "model", "config.json")):
            base_path = os.path.join(model_path, "model")

        self.processor = Qwen2_5OmniProcessor.from_pretrained(base_path)

        last_err = None
        for impl in ([attn] if attn else ["flash_attention_2", "sdpa"]):
            try:
                self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                    base_path, torch_dtype=torch.bfloat16, device_map=device,
                    attn_implementation=impl)
                break
            except (ImportError, ValueError) as e:
                last_err = e
        else:
            raise last_err

        lora_path = lora_path or self._find_lora(model_path)
        if lora_path:
            if isinstance(lora_path, tuple):  # (hub repo, subfolder)
                self.model = PeftModel.from_pretrained(self.model, lora_path[0], subfolder=lora_path[1])
            else:
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"loaded LoRA adapter from {lora_path}")
        else:
            print("no LoRA adapter found, evaluating the base model as-is")
        self.model.eval()

        # Discover the probability tokens from the vocab so token-count
        # ablations (e.g. 20 or 50 bins) evaluate without extra flags.
        vocab = self.processor.tokenizer.get_vocab()
        conf = sorted(
            (int(m.group(1)), tid)
            for tok, tid in vocab.items()
            if (m := re.fullmatch(r"<CON_(\d+)>", tok))
        )
        if not conf:
            raise RuntimeError("tokenizer has no <CON_i> probability tokens — wrong model?")
        self.conf_ids = torch.tensor([tid for _, tid in conf])
        self.scale = torch.linspace(0.0, 1.0, len(conf))
        print(f"found {len(conf)} probability tokens")

    @staticmethod
    def _find_lora(model_path: str):
        if os.path.isdir(model_path):
            local = os.path.join(model_path, "lora")
            if os.path.exists(os.path.join(local, "adapter_config.json")):
                return local
            return None
        from huggingface_hub import file_exists
        if file_exists(model_path, "lora/adapter_config.json"):
            return (model_path, "lora")
        return None

    def _prepare(self, batch_messages: list, modality: str, use_audio_in_video: bool):
        text = self.processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=False)
        if modality == "text":
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
        else:
            audios, images, videos = self.process_mm_info(
                batch_messages, use_audio_in_video=use_audio_in_video)
            inputs = self.processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True,
                use_audio_in_video=use_audio_in_video)
        return inputs.to(self.model.device).to(self.model.dtype)

    def score(self, batch_messages: list, modality: str) -> list:
        """Return one probability in [0, 1] per message: the expected value of
        the <CON_i> distribution at the first generated position."""
        torch = self.torch
        with torch.no_grad():
            if modality == "omni":
                try:
                    inputs = self._prepare(batch_messages, modality, use_audio_in_video=True)
                except Exception:  # some videos have no audio track
                    inputs = self._prepare(batch_messages, modality, use_audio_in_video=False)
            else:
                inputs = self._prepare(batch_messages, modality, use_audio_in_video=(modality == "audio"))

            logits = self.model(**inputs).logits  # (B, L, V)

            # last non-pad position of each sample, valid for either padding side
            mask = inputs["attention_mask"]
            last = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
            final = logits[torch.arange(logits.shape[0], device=logits.device), last]

            conf_logits = final[:, self.conf_ids.to(logits.device)].float()
            probs = torch.softmax(conf_logits, dim=-1)
            scores = probs @ self.scale.to(logits.device)
            result = scores.cpu().tolist()
        torch.cuda.empty_cache()
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() > 0:
            total += m.sum() / len(y) * abs(np.mean(y[m]) - np.mean(p[m]))
    return float(total)


def nll(y, p) -> float:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-10, 1 - 1e-10)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def krippendorff_alpha(human: list, model: list, bins: int = 10):
    """Interval-level alpha on RIDIT-scored bins (same recipe as src/analyse)."""
    try:
        import krippendorff
    except ImportError:
        print("note: `krippendorff` package not installed, alpha reported as null")
        return None
    binned = np.floor(np.array([human, model]) * bins).astype(int)
    binned[binned == bins] = bins - 1
    flat = binned.flatten()
    counts = np.bincount(flat - flat.min())
    props = counts[counts > 0] / len(flat)
    cumdist = np.cumsum(props)
    ridit_scores = np.insert(cumdist[:-1], 0, 0.0) + props / 2.0
    ridit_map = dict(zip(np.unique(flat), ridit_scores))
    ridit_data = [[ridit_map[v] for v in row] for row in binned]
    return float(krippendorff.alpha(reliability_data=ridit_data, level_of_measurement="interval"))


def compute_metrics(records: list) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    labels = [int(r["label"]) for r in records]
    probs = [r["prob"] for r in records]
    preds = [int(p > THRESHOLD) for p in probs]
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    # Soft-target metrics are only meaningful against a human probability
    # (matching the original eval_mse); they are computed on the annotated
    # subset, binary-only datasets get acc/F1 only.
    with_p = [r for r in records if r.get("probability") is not None]
    if with_p:
        human = [r["probability"] for r in with_p]
        model = [r["prob"] for r in with_p]
        metrics["nll"] = nll(human, model)
        metrics["ece"] = ece(human, model)
        metrics["mse"] = float(np.mean((np.array(human) - np.array(model)) ** 2))
        metrics["krippendorff_alpha"] = krippendorff_alpha(human, model)
    return metrics


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def item_key(item: dict) -> tuple:
    return (item.get("path", ""), item.get("sentence", ""), item["claim"])


def read_jsonl(path: str) -> list:
    records = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo or local dir (base model; `lora/` subfolder auto-detected)")
    parser.add_argument("--lora", default=None, help="explicit LoRA adapter path, overrides auto-detection")
    parser.add_argument("--dataset", required=True, choices=["unli", "clotho", "wikivideo"])
    parser.add_argument("--modality", default=None, choices=["text", "audio", "video", "omni"],
                        help="defaults: unli=text, clotho=audio, wikivideo=video")
    parser.add_argument("--data-repo", default=DEFAULT_DATA_REPO)
    parser.add_argument("--data-file", default=None,
                        help="local label file, skips the default source (wikivideo default: "
                             "<media-root>/annotations/final_data_2015-2025.json)")
    parser.add_argument("--annotation-file", default=None,
                        help="wikivideo only: local UMUI-annotation/annotation.json with human probabilities")
    parser.add_argument("--split", default="validation", help="unli only")
    parser.add_argument("--media-root", default=os.environ.get("UMUI_MEDIA_ROOT"),
                        help="directory holding the extracted media tarballs (env: UMUI_MEDIA_ROOT)")
    parser.add_argument("--output-dir", default="output/eval")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn", default=None, help="attention implementation; default tries flash_attention_2 then sdpa")
    parser.add_argument("--start", type=int, default=0, help="dataset slice start (for slurm sharding)")
    parser.add_argument("--end", type=int, default=-1, help="dataset slice end, -1 = all")
    parser.add_argument("--metrics-only", action="store_true", help="recompute metrics from the existing output file")
    args = parser.parse_args()

    if args.modality is None:
        args.modality = {"unli": "text", "clotho": "audio", "wikivideo": "video"}[args.dataset]
    if args.dataset == "unli" and args.modality != "text":
        raise SystemExit("unli is text-only")
    if args.dataset == "clotho" and args.modality != "audio":
        raise SystemExit("clotho is audio-only")

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f"{args.dataset}_{args.modality}")
    output_path = prefix + ".jsonl"
    metrics_path = prefix + ".metrics.json"

    if args.metrics_only:
        records = read_jsonl(output_path)
        if not records:
            raise SystemExit(f"no predictions found at {output_path}")
    else:
        from tqdm import tqdm

        items = load_items(args)
        done = {item_key(r) for r in read_jsonl(output_path)}
        todo = [item for item in items if item_key(item) not in done]

        if todo:
            scorer = UNLIScorer(args.model, args.lora, args.device, args.attn)
            with open(output_path, "a") as out:
                for i in tqdm(range(0, len(todo), args.batch_size)):
                    batch = todo[i:i + args.batch_size]
                    messages = [build_messages(item, args.modality) for item in batch]
                    scores = scorer.score(messages, args.modality)
                    for item, prob in zip(batch, scores):
                        record = {**item, "prob": prob, "answer": f"<answer>{prob}</answer>"}
                        out.write(json.dumps(record) + "\n")
                    out.flush()
        records = read_jsonl(output_path)

    metrics = compute_metrics(records)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))
    print(f"predictions: {output_path}\nmetrics: {metrics_path}")


if __name__ == "__main__":
    main()
