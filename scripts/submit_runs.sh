#!/bin/bash
# Submits the runs behind the paper's table. Run ON THE GPU LOGIN NODE (trig-login01);
# Trillium rejects GPU jobs from other nodes.
#
#   bash scripts/submit_runs.sh c0   # the direct judge
#   bash scripts/submit_runs.sh c1   # MARCH
#
#   SCOPE=pilot (default): gemini, three tiers, Qwen3-8B            (24 shards)
#   SCOPE=full           : 9 CodeJudgeBench cells + HumanEvalFix, both judges (148 shards)
#
# Outputs go to $STUDY/march-gens-<WHAT>/ (one directory per cell, shard_N.jsonl inside). Then:
#   python3 scripts/make_table.py --c0 <march-gens-c0> --c1 <march-gens-c1> [--hef]
set -euo pipefail
WHAT="${1:?usage: submit_runs.sh c0|c1}"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
STUDY="${STUDY:-/scratch/salmayg/multi-agent-study}"      # holds hf/models/Qwen3-8B and Qwen3-14B
case "$WHAT" in
  c0) COND=c0_direct ;;
  c1) COND=c1_verbatim ;;
  *) echo "unknown: $WHAT"; exit 2 ;;
esac
GENS="${GENS_ROOT:-$STUDY/march-gens-$WHAT}"
ACCT="${ACCT:-def-assafh}"; EXCLUDE="${EXCLUDE:-trig0018,trig0012}"; SEED="${SEED:-1234}"
M8="$STUDY/hf/models/Qwen3-8B"; M14="$STUDY/hf/models/Qwen3-14B"
mkdir -p "$GENS"; cd "$PKG/pipeline"                     # xfer_cell.sh runs $PROJ/src/run_condition.py
common="ALL,PROJ=$PKG/pipeline,CONDITION=$COND,TASK=codegen,LIMIT=0,USE_ZTR=1,SEED=$SEED,GENS_ROOT=$GENS"
sub () {  # dataset split tier judge nshards [extra exports]
  local ds=$1 split=$2 tier=$3 judge=$4 nsh=$5 extra="${6:-}" model=$M8 jtag="" wall=04:00:00
  [ "$judge" = 14b ] && { model=$M14; jtag=qwen3-14b; wall=06:00:00; }
  [ "$ds" = humanevalfix ] && wall=06:00:00
  local name="${WHAT}_${split}_${tier}_${judge}"
  local jid; jid=$(sbatch --parsable --account="$ACCT" --partition=compute --time="$wall" --exclude="$EXCLUDE" \
    --requeue --array=0-$((nsh-1)) --job-name="$name" \
    --export="$common,DATASET=$ds,SPLIT=$split,DIFFICULTY=$tier,NSHARDS=$nsh,MODEL=$model,JUDGETAG=$jtag,${extra}X=X" \
    slurm/xfer_cell.sh)
  echo "submitted $jid  $name"
}
if [ "${SCOPE:-pilot}" = full ]; then
  for gen in claude_3.7_sonnet gemini_2.5_pro qwen3_235b; do for tier in easy medium hard; do
    sub codejudgebench $gen $tier 8b 8; sub codejudgebench $gen $tier 14b 8; done; done
  for judge in 8b 14b; do sub humanevalfix hef all $judge 4 "LOCAL_JSONL=$PKG/data/hef_pairs.jsonl,"; done
else
  for tier in easy medium hard; do sub codejudgebench gemini_2.5_pro $tier 8b 8; done
fi
echo "outputs -> $GENS"
