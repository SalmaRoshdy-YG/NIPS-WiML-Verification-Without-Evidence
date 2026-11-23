#!/usr/bin/env python
"""
Offline logic tests. No GPU, no model, no network -- runs in ~1s on a login node.

    python src/selftest.py

Covers the parts that would otherwise only fail after a queue wait:
  * MARCH's verbatim prompts are byte-identical to the release (if the source
    tree is present alongside)
  * numeric vs verdict answer comparison
  * ZTR scoring including the zero-claim padding path
  * the full Solver->Proposer->Checker pipeline against a mocked LLM
  * pointwise -> pairwise conversion and label/pred polarity
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub the heavy deps so this runs anywhere.
for _n in ("vllm", "transformers", "datasets"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        sys.modules[_n] = _m
if not hasattr(sys.modules["vllm"], "LLM"):
    sys.modules["vllm"].LLM = object
    sys.modules["vllm"].SamplingParams = object
if not hasattr(sys.modules["transformers"], "AutoTokenizer"):
    sys.modules["transformers"].AutoTokenizer = object
if not hasattr(sys.modules["datasets"], "load_dataset"):
    sys.modules["datasets"].load_dataset = lambda *a, **k: None

import march_verbatim as MV          # noqa: E402
import march_code as MC              # noqa: E402
import run_condition as RC           # noqa: E402
from types import SimpleNamespace    # noqa: E402

FAILED = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------- primitives
print("=== MARCH primitives ===")
check("extract_qa_pairs numeric",
      MV.extract_qa_pairs("- Question: How many? [Answer: 42]")[0]["answer"] == "42")
check("extract_qa_pairs is answer-agnostic (C2 reuses it)",
      MV.extract_qa_pairs("- Question: ok? [Answer: CORRECT]")[0]["answer"] == "CORRECT")
check("extract_detailed_info",
      MV.extract_detailed_info("1. Evidence: e [Answer: 7]")[1]["answer"] == "7")
check("check_ans_eq numeric equal", MV.check_ans_eq("42", "42"))
check("check_ans_eq strips %/k/commas", MV.check_ans_eq("1,000", "1000"))
check("check_ans_eq float tolerance", MV.check_ans_eq("3.14", "3.15"))
check("check_ans_eq rejects mismatch", not MV.check_ans_eq("1", "2"))
check("check_verdict_eq agreement", MC.check_verdict_eq("CORRECT", "CORRECT"))
check("check_verdict_eq UNKNOWN never agrees",
      not MC.check_verdict_eq("UNKNOWN", "UNKNOWN"))
check("check_verdict_eq rejects mismatch",
      not MC.check_verdict_eq("CORRECT", "INCORRECT"))

print("=== ZTR ===")
check("ztr zero claims -> 0.0 (padding path)", MV.ztr_score(0, 0) == 0.0)
check("ztr any fault -> -1.0", MV.ztr_score(1, 3) == -1.0)
check("ztr clean -> 0.0", MV.ztr_score(0, 3) == 0.0)
check("error-rate variant", abs(MV.ztr_score(2, 5, use_ztr=False) - (-0.2)) < 1e-9)

# --------------------------------------------------------------- CJB parsing
print("=== CodeJudgeBench parsing ===")
check("verdict A", RC.extract_pairwise_pred("x [[A>B]]", "stop") == "A")
check("verdict B", RC.extract_pairwise_pred("<think>t</think> [[B>A]]", "stop") == "B")
check("truncated -> no verdict", RC.extract_pairwise_pred("[[A>B]]", "length") == "")
check("no marker -> no verdict", RC.extract_pairwise_pred("unsure", "stop") == "")

# ------------------------------------------------------------------ pipeline
print("=== pipeline (mocked LLM) ===")
ITEMS = [
    {"_uid": "t0", "difficulty": "hard", "question_content": "Sum a list.",
     "pos_response": "def s(x): return sum(x)",
     "neg_response": "def s(x): return sum(x[1:])"},
    {"_uid": "t1", "difficulty": "medium", "question_content": "Max of list.",
     "pos_response": "def m(x): return max(x)",
     "neg_response": "def m(x): return x[0]"},
]
ARGS = SimpleNamespace(use_ztr=True)


class MockChat:
    """call 1 = Solver, 2 = Proposer, 3 = Checker (n=3)."""
    def __init__(self, proposer, checker):
        self.calls = 0
        self.p, self.c = proposer, checker

    def __call__(self, prompts):
        self.calls += 1
        if self.calls == 1:
            return [[("analysis FINAL: CORRECT", "stop")] for _ in prompts]
        if self.calls == 2:
            return [[(self.p, "stop")] for _ in prompts]
        return [[(self.c, "stop")] * 3 for _ in prompts]


# C1: the numeric Proposer emits nothing parseable on code.
r = RC.run_march(ITEMS, "codegen",
                 MockChat("No numeric facts present.", "1. Evidence: n [Answer: 0]"),
                 "c1_verbatim", ARGS, None)
check("C1 rows = 2 x items", len(r) == 4)
check("C1 zero claims", r[0]["pos_n_claims"] == 0)
check("C1 padding path taken", r[0]["pos_padding"] is True)
check("C1 all ties", all(x["tie"] for x in r))
check("C1 no verdict emitted", all(x["pred"] == "" for x in r))
check("C1 score 0.0 both sides", r[0]["pos_score"] == 0.0 and r[0]["neg_score"] == 0.0)

PROP = ("- Question: Sums all elements? [Answer: CORRECT]\n"
        "- Question: Handles empty? [Answer: CORRECT]\n"
        "- Question: Off by one? [Answer: INCORRECT]")

# C2 clean: Checker agrees on every claim.
r = RC.run_march(ITEMS, "codegen",
                 MockChat(PROP,
                          "1. Evidence: a [Answer: CORRECT]\n"
                          "2. Evidence: b [Answer: CORRECT]\n"
                          "3. Evidence: c [Answer: INCORRECT]"),
                 "c2_adapted", ARGS, None)
check("C2 claims parsed", r[0]["pos_n_claims"] == 3)
check("C2 no padding", r[0]["pos_padding"] is False)
check("C2 no faults", r[0]["pos_fault"] == 0)
check("C2 score 0.0", r[0]["pos_score"] == 0.0)

# C2 with disagreement: ZTR fires.
r = RC.run_march(ITEMS, "codegen",
                 MockChat(PROP,
                          "1. Evidence: a [Answer: INCORRECT]\n"
                          "2. Evidence: b [Answer: CORRECT]\n"
                          "3. Evidence: c [Answer: INCORRECT]"),
                 "c2_adapted", ARGS, None)
check("C2 fault detected", r[0]["pos_fault"] == 1)
check("C2 ZTR fires", r[0]["pos_score"] == -1.0)

# Polarity: both orderings present, labels correct.
fwd = [x for x in r if not x["reverse"]]
rev = [x for x in r if x["reverse"]]
check("forward rows labelled A", all(x["label"] == "A" for x in fwd))
check("reverse rows labelled B", all(x["label"] == "B" for x in rev))
check("both orderings emitted", len(fwd) == len(rev) == 2)

# Asymmetric scores must break the tie and map to the right side.
class Asym:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompts):
        self.calls += 1
        if self.calls == 1:
            return [[("a FINAL: CORRECT", "stop")] for _ in prompts]
        if self.calls == 2:
            return [[(PROP, "stop")] for _ in prompts]
        # units are ordered [item0-pos, item0-neg, item1-pos, item1-neg]
        good = ("1. Evidence: a [Answer: CORRECT]\n"
                "2. Evidence: b [Answer: CORRECT]\n"
                "3. Evidence: c [Answer: INCORRECT]")
        bad = ("1. Evidence: a [Answer: INCORRECT]\n"
               "2. Evidence: b [Answer: CORRECT]\n"
               "3. Evidence: c [Answer: INCORRECT]")
        return [[((good if i % 2 == 0 else bad), "stop")] * 3
                for i in range(len(prompts))]


r = RC.run_march(ITEMS, "codegen", Asym(), "c2_adapted", ARGS, None)
f0 = [x for x in r if x["uid"] == "t0" and not x["reverse"]][0]
r0 = [x for x in r if x["uid"] == "t0" and x["reverse"]][0]
check("asymmetric scores break tie", f0["tie"] is False)
check("pos favoured -> forward pred A", f0["pred"] == "A")
check("pos favoured -> reverse pred B", r0["pred"] == "B")
check("forward row is correct", f0["pred"] == f0["label"])
check("reverse row is correct", r0["pred"] == r0["label"])

# ------------------------------------------------------------ spawn safety
print("=== vLLM spawn safety ===")
_here = os.path.dirname(os.path.abspath(__file__))
for _entry in ("run_condition.py", "pilot.py"):
    _p = os.path.join(_here, _entry)
    if not os.path.exists(_p):
        print(f"  SKIP  {_entry} absent")
        continue
    _src = open(_p).read()
    check(f"{_entry} guards __main__", 'if __name__ == "__main__":' in _src)
    # Heavy imports at module top level get re-executed in every spawned
    # worker. transformers/datasets/vllm must be imported inside functions.
    _top = [ln for ln in _src.splitlines()
            if (ln.startswith("import ") or ln.startswith("from "))
            and any(m in ln for m in ("vllm", "transformers", "datasets"))]
    check(f"{_entry} defers heavy imports", not _top)

_sl = os.path.join(_here, "..", "slurm")
if os.path.isdir(_sl):
    for _f in sorted(os.listdir(_sl)):
        if not _f.endswith(".sh"):
            continue
        _s = open(os.path.join(_sl, _f)).read()
        # `python - <<EOF` makes __main__ == '<stdin>', which spawn cannot
        # re-import; workers die with FileNotFoundError.
        check(f"{_f} has no piped-stdin python driver",
              "python - <<" not in _s)
        check(f"{_f} does not srun-wrap python",
              not any(ln.strip().startswith("srun python")
                      for ln in _s.splitlines()))


for cand in ("../MARCH-main/quarl/quarl/trainer/ray_trainer_with_fact_check_march.py",
             "MARCH/quarl/quarl/trainer/ray_trainer_with_fact_check_march.py"):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", cand)
    if os.path.exists(p):
        src = open(p).read()
        q = src.split('question_prompt = """')[1].split('"""')[0]
        a = src.split('answer_question_prompt = """')[1].split('"""')[0]
        check("question_prompt byte-identical", q == MV.question_prompt)
        check("answer_question_prompt byte-identical", a == MV.answer_question_prompt)
        break
else:
    print("  SKIP  MARCH source tree not found alongside; nothing to diff")

# ---------------------------------------------------- trace harness (C4)
# Delegated to xfer_smoke_test.py, which needs subprocess execution and takes
# ~15s. Skipped by default so the core checks stay instant; run with
# `python src/selftest.py --full` or `python src/xfer_smoke_test.py` directly.
if "--full" in sys.argv:
    print("\n=== trace harness (delegated) ===")
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(_here, "xfer_smoke_test.py")],
        capture_output=True, text=True,
    )
    n_pass = r.stdout.count("PASS")
    check(f"xfer_smoke_test ({n_pass} checks)", r.returncode == 0)
    if r.returncode != 0:
        print(r.stdout[-1500:])
else:
    print("\n=== trace harness ===")
    print("  SKIP  run with --full, or: python src/xfer_smoke_test.py")

print()
if FAILED:
    print(f"{len(FAILED)} FAILURE(S):")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("ALL TESTS PASSED")
