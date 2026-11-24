"""Pre-download the model and the MBPP+ dataset on a node WITH internet access
(login node). Compute nodes on most HPC clusters are offline, so run this once
before submitting the sbatch job.

Usage:
    python download_assets.py --model_name Qwen/Qwen2.5-Coder-3B-Instruct
"""
import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download


def download_assets(model_name: str) -> None:
    """Download model weights and the MBPP+ dataset into the local HF cache.

    Input:  model_name (str) - Hugging Face model id.
    Output: None (files are stored in the HF cache, i.e. $HF_HOME).
    """
    print(f"Downloading model: {model_name}")
    snapshot_download(repo_id=model_name)
    print("Downloading dataset: evalplus/mbppplus")
    load_dataset("evalplus/mbppplus", split="test")
    print("Done.")


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parsed_arguments = argument_parser.parse_args()
    download_assets(parsed_arguments.model_name)
