#!/bin/bash
#SBATCH --job-name=self_correcting_agent
#SBATCH --partition=gpu                 # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="H200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11                    # 3 prompt styles x 4 shards = 12 array tasks
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Before submitting:
#   python prepare_data.py              (login node, once; creates data/raw, data/processed, data/private)
#   mkdir -p logs results

set -euo pipefail

# ---- Environment (edit for your cluster) ----
# module load cuda/12.1 python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate coding_agent
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1                 # compute nodes are usually offline; run download_assets.py first
export TOKENIZERS_PARALLELISM=false

# ---- Experiment settings ----
MODEL_NAME="Qwen/Qwen2.5-Coder-3B-Instruct"
SPLIT="dev"                             # "dev" (90%) for main runs; "eval" (held-out 10%) for the final run
NUM_PROBLEMS=200
USE_ALL_TESTS=0                         # 1 = also run the extra use_all_tests condition
NUM_SHARDS=4
PROMPT_STYLES=(base cot react)          # results go to results/base, results/cot, results/react

PROMPT_STYLE=${PROMPT_STYLES[$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))]}
SHARD_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))

EXTRA_FLAGS=""
if [ "${USE_ALL_TESTS}" = "1" ]; then EXTRA_FLAGS="--use_all_tests"; fi

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Node: $(hostname) | style ${PROMPT_STYLE} | shard ${SHARD_INDEX}/${NUM_SHARDS} | split ${SPLIT}"
nvidia-smi || true

python run_experiment.py \
    --model_name "${MODEL_NAME}" \
    --prompt_style "${PROMPT_STYLE}" \
    --split "${SPLIT}" \
    --num_problems ${NUM_PROBLEMS} \
    --max_rounds 5 \
    --reflection_rounds 2 \
    --shard_index ${SHARD_INDEX} \
    --num_shards ${NUM_SHARDS} \
    --data_directory data \
    --results_root results \
    ${EXTRA_FLAGS}

echo "Finished style ${PROMPT_STYLE} shard ${SHARD_INDEX}"
