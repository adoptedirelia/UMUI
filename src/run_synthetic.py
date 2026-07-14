
import os
import json
import multiprocessing as mp

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from src.synthetic_data import parse_config, SyntheticDataGenerator


def main() -> None:
    config = parse_config()
    # dataset_loader = DatasetLoader(config)
    # dataset_loader.load()

    generator = SyntheticDataGenerator(config)


    results = generator.run_batch()


    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    # Keep the same start method expectation as original
    mp.set_start_method("spawn", force=True)
    main()