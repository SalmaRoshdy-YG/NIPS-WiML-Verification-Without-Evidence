#!/bin/bash
# =============================================================================
# One (condition, task, split) cell as a Slurm array; each task = one shard.
# Cluster: Rorqual (H100 80GB)
#
# Submit via scripts/xfer_submit_grid.sh, or directly:
#   sbatch --export=ALL,CONDITION=c1_verbatim,TASK=codegen,SPLIT=gemini_2.5_pro \
#          slurm/xfer_cell.sh
#
# Qwen3-8B in bf16 is ~16GB; on an 80GB H100 one GPU is ample and leaves room
# for the 40k context needed by thinking mode. Do NOT request more than 1 GPU:
# tensor_parallel_size is inferred from CUDA_VISIBLE_DEVICES and TP>1 buys
# nothing at this model size while queueing far worse.
# =============================================================================
#SBATCH --account=def-kobti
#SBATCH --job-name=xfer
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:59:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/salmayg/multi-agent-study/xferlogs/%x-%A_%a.out

set -euo pipefail

: "${CONDITION:?set CONDITION}"
: "${TASK:?set TASK}"
: "${SPLIT:?set SPLIT}"
DATASET="${DATASET:-codejudgebench}"
DIFFICULTY="${DIFFICULTY:-all}"
LIMIT="${LIMIT:-0}"          # 0 = whole tier
USE_ZTR="${USE_ZTR:-1}"      # 0 = error-rate reward (C1-ER / C2-ER)
NSHARDS="${NSHARDS:-8}"
SEED="${SEED:-1234}"        # 1234 = the reported results

PROJ="${PROJ:-$SLURM_SUBMIT_DIR}"
VENV="${VENV:-$SCRATCH/multi-agent-study/march_venv}"

export HF_HOME=$SCRATCH/multi-agent-study/hf
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_STREAMING=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# keep every cache off $HOME -- its quota is small and will kill the job
export XDG_CACHE_HOME=$SCRATCH/multi-agent-study/cache
export TMPDIR=${SLURM_TMPDIR:-$SCRATCH/multi-agent-study/tmp}
export OUTLINES_CACHE_DIR=$TMPDIR/outlines
export TRITON_CACHE_DIR=$TMPDIR/triton
export VLLM_CACHE_ROOT=$TMPDIR/vllm
export FLASHINFER_WORKSPACE_BASE=$SCRATCH/multi-agent-study/cache
# vLLM writes usage stats under ~/.config; that failed while $HOME was full
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
mkdir -p "$XDG_CACHE_HOME" "$OUTLINES_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

module purge
module load StdEnv/2023 gcc/12.3 python/3.11 arrow/25.0.0 cuda/12.2 opencv
source "$VENV/bin/activate"

TAG=""
[ "$USE_ZTR" = "0" ] && TAG="-er"
# Seed 1234 keeps its original path so the reported results are
# untouched; every other seed gets its own directory.
SEEDTAG=""
[ "$SEED" != "1234" ] && SEEDTAG="__s${SEED}"
CELL="${CONDITION}${TAG}__${DATASET}__${TASK}__${SPLIT}__${DIFFICULTY}${JUDGETAG:+__$JUDGETAG}${SEEDTAG}"
OUTDIR=${GENS_ROOT:-$SCRATCH/multi-agent-study/march-gens}/$CELL

mkdir -p "$OUTDIR" "$SCRATCH/multi-agent-study/march-logs"


TRACES_ARG=""
if [ "$CONDITION" = "c4_exec" ]; then
  TRACES="$SCRATCH/multi-agent-study/march-traces/${TASK}__${SPLIT}__${DIFFICULTY}.jsonl"
  [ -f "$TRACES" ] || { echo "[fatal] c4_exec needs $TRACES" >&2; exit 1; }
  TRACES_ARG="--traces $TRACES"
fi



echo "[cell] $CELL shard $SLURM_ARRAY_TASK_ID / $NSHARDS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# No srun: vLLM manages its own worker processes via multiprocessing spawn.
# Wrapping in srun adds a second layer of process management that interferes
# with the spawn bootstrap.
python $PROJ/src/run_condition.py \
  --condition "$CONDITION" \
  --dataset   "$DATASET" \
  --task      "$TASK" \
  --split     "$SPLIT" \
  --difficulty "$DIFFICULTY" \
  --limit "$LIMIT" \
  --use-ztr "$USE_ZTR" \
  --model     "${MODEL:-$SCRATCH/multi-agent-study/hf/models/Qwen3-8B}" \
  --shard     "$SLURM_ARRAY_TASK_ID" \
  --nshards   "$NSHARDS" \
  --out       "$OUTDIR/shard_${SLURM_ARRAY_TASK_ID}.jsonl" \
  --seed      "$SEED" \
  --max-tokens "${MAX_TOKENS:-32768}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --gpu-mem-util 0.92 \
  ${LOCAL_JSONL:+--local-jsonl "$LOCAL_JSONL"} \
  $TRACES_ARG

echo "[ok] $CELL shard $SLURM_ARRAY_TASK_ID"
