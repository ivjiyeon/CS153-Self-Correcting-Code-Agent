"""Pre-download the model and the MBPP+ data on a node WITH internet access
(login node). Compute nodes on most HPC clusters are offline.

Usage:
    python download_assets.py --model_name Qwen/Qwen2.5-Coder-3B-Instruct
Then run `python prepare_data.py` once to build data/raw, data/processed, data/private.
"""
import argparse

from evalplus.data import get_mbpp_plus
from huggingface_hub import snapshot_download


def download_assets(model_name: str) -> None:
    """Download model weights and the MBPP+ data into local caches.

    Input:  model_name (str) - Hugging Face model id.
    Output: None (weights go to $HF_HOME, MBPP+ to the EvalPlus cache).
    """
    print(f"Downloading model: {model_name}")
    snapshot_download(repo_id=model_name)
    print("Downloading MBPP+ through evalplus.data.get_mbpp_plus()")
    problem_count = len(get_mbpp_plus())
    print(f"Done. MBPP+ problems: {problem_count}")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parsed_arguments = argument_parser.parse_args()
    download_assets(parsed_arguments.model_name)