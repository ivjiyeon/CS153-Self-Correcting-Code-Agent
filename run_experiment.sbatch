#!/bin/bash
#SBATCH --job-name=self_correcting_agent
#SBATCH --partition=gpu                 # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-3                     # 4 shards, one GPU each; use 0-0 for a single job
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Before submitting:  mkdir -p logs results

set -euo pipefail

# ---- Environment (edit for your cluster) ----
# module load cuda/12.1 python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate coding_agent
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1                 # compute nodes are usually offline; run download_assets.py first
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="Qwen/Qwen2.5-Coder-3B-Instruct"
NUM_PROBLEMS=200
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Node: $(hostname) | shard ${SHARD_INDEX}/${NUM_SHARDS}"
nvidia-smi || true

python run_experiment.py \
    --model_name "${MODEL_NAME}" \
    --num_problems ${NUM_PROBLEMS} \
    --num_visible_tests 3 \
    --max_rounds 5 \
    --reflection_rounds 2 \
    --shard_index ${SHARD_INDEX} \
    --num_shards ${NUM_SHARDS} \
    --output_dir results

echo "Finished shard ${SHARD_INDEX}"
