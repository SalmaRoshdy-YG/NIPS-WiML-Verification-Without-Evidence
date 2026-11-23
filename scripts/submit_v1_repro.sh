#!/bin/bash
# Reruns MARCH (C1) with the v1 pipeline, defect included, to regenerate the numbers in the
# arXiv v1 abstract. Run it ON THE GPU LOGIN NODE (trig-login01); Trillium rejects GPU jobs
# from other nodes. Nothing here touches the corrected results.
#
#   bash scripts/submit_v1_repro.sh            # pilot: gemini, three tiers, Qwen3-8B (24 shards)
#   SCOPE=full bash scripts/submit_v1_repro.sh # all nine CodeJudgeBench cells, both judges
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
STUDY="${STUDY:-/scratch/salmayg/multi-agent-study}"
GENS="${GENS_ROOT:-$STUDY/march-gens-v1repro}"      # separate from every other result directory
ACCT="${ACCT:-def-assafh}"; EXCLUDE="${EXCLUDE:-trig0018,trig0012}"; SEED="${SEED:-1234}"
M8="$STUDY/hf/models/Qwen3-8B"; M14="$STUDY/hf/models/Qwen3-14B"
mkdir -p "$GENS"; cd "$PKG/v1_pipeline"      # xfer_cell.sh runs $PROJ/src/run_condition.py
sub () {  # gen tier judge
  local gen=$1 tier=$2 judge=$3 model=$M8 jtag="" wall=04:00:00
  [ "$judge" = 14b ] && { model=$M14; jtag=qwen3-14b; wall=06:00:00; }
  local name="v1repro_${gen}_${tier}_${judge}"
  local jid; jid=$(sbatch --parsable --account="$ACCT" --partition=compute --time="$wall" --exclude="$EXCLUDE" \
    --requeue --array=0-7 --job-name="$name" \
    --export=ALL,PROJ="$PKG/v1_pipeline",CONDITION=c1_verbatim,DATASET=codejudgebench,TASK=codegen,SPLIT="$gen",DIFFICULTY="$tier",LIMIT=0,NSHARDS=8,USE_ZTR=1,SEED="$SEED",MODEL="$model",JUDGETAG="$jtag",GENS_ROOT="$GENS",X=X \
    slurm/xfer_cell.sh)
  echo "submitted $jid  $name"
}
if [ "${SCOPE:-pilot}" = full ]; then
  for gen in claude_3.7_sonnet gemini_2.5_pro qwen3_235b; do for tier in easy medium hard; do
    sub $gen $tier 8b; sub $gen $tier 14b; done; done
  # HumanEvalFix: 142 pairs from data/hef_pairs.jsonl, 4 shards per cell
  for judge in 8b 14b; do
    model=$M8; jtag=""; [ "$judge" = 14b ] && { model=$M14; jtag=qwen3-14b; }
    jid=$(sbatch --parsable --account="$ACCT" --partition=compute --time=06:00:00 --exclude="$EXCLUDE" \
      --requeue --array=0-3 --job-name="v1repro_hef_$judge" \
      --export=ALL,PROJ="$PKG/v1_pipeline",CONDITION=c1_verbatim,DATASET=humanevalfix,TASK=codegen,SPLIT=hef,DIFFICULTY=all,LIMIT=0,NSHARDS=4,USE_ZTR=1,SEED="$SEED",MODEL="$model",JUDGETAG="$jtag",GENS_ROOT="$GENS",LOCAL_JSONL="$PKG/data/hef_pairs.jsonl",X=X \
      slurm/xfer_cell.sh)
    echo "submitted $jid  v1repro_hef_$judge"
  done
else
  for tier in easy medium hard; do sub gemini_2.5_pro $tier 8b; done
fi
echo "outputs -> $GENS ; when done: python3 scripts/make_table.py --c0 $STUDY/march-gens-trillium --c1 $GENS"
