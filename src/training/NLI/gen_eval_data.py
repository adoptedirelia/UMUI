"""Generate eval-data JSON files from evaldataset.py.

Produces one JSON file per (dataset, modality):
  wikivideo_video, wikivideo_audio, wikivideo_av, unli, clotho

Each file is a list of samples ({claim, label, modality, path/sentence, ...})
as produced by the dataset classes' `processed_data`.
"""
import os
import json

from config import OmniEvalConfig
from evaldataset import WikiVideoDataset, UNLI_Dataset, ClothoDataset

# Actual on-disk data locations (differ from config defaults).
DATA_BASE = "/exp/dzhang1/UMUI"
OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "eval_data")


def make_config():
    cfg = OmniEvalConfig()
    # wikivideo
    cfg.wikivideo_pre_path = os.path.join(DATA_BASE, "wikivideo_/data/combined_videos")
    cfg.wikivideo_audio_pre_path = os.path.join(DATA_BASE, "wikivideo_/data/audios/en")
    cfg.wikivideo_label_path = os.path.join(DATA_BASE, "wikivideo_/annotations/final_data_2015-2025.json")
    # clotho
    cfg.clotho_pre_path = os.path.join(DATA_BASE, "audio_entailment/clotho")
    cfg.clotho_label_path = os.path.join(DATA_BASE, "AudioEntailment/data/CLE")
    # unli
    cfg.unli_label_path = os.path.join(DATA_BASE, "training-data/unli")
    return cfg


def flatten(data):
    """WikiVideoDataset returns {event: [items]}; flatten to a list."""
    if isinstance(data, dict):
        return [item for items in data.values() for item in items]
    return data


def dump(name, data):
    flat = flatten(data)
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)
    n_true = sum(1 for x in flat if x["label"])
    print(f"{name:18s} -> {path}  (n={len(flat)}, True={n_true}, False={len(flat) - n_true})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # wikivideo: video / audio / av(=omni)
    for out_name, modality in [
        ("wikivideo_video", "video"),
        ("wikivideo_audio", "audio"),
        ("wikivideo_av", "omni"),
    ]:
        cfg = make_config()
        cfg.modality = modality
        dump(out_name, WikiVideoDataset(cfg).processed_data)

    # unli (text)
    cfg = make_config()
    dump("unli", UNLI_Dataset(cfg).processed_data)

    # clotho (audio)
    cfg = make_config()
    dump("clotho", ClothoDataset(cfg).processed_data)


if __name__ == "__main__":
    main()
