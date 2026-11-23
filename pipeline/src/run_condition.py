#!/usr/bin/env python
"""
Run one (condition, task, dataset, split, shard) cell and write a JSONL shard.

Conditions
----------
c0_direct    CodeJudgeBench pairwise judge, verbatim prompt + parsing.
c1_verbatim  MARCH exactly as published: numeric Proposer, numeric Checker,
             check_ans_eq, ZTR. Expected to collapse; that collapse is the
             headline result.
c2_adapted   MARCH with behavioural claims (blinded Checker).
c3_unblinded C2 with the Solver's analysis appended to the Checker's context.
c4_exec      C2 with execution traces as the Checker's evidence.

All MARCH conditions are pointwise: the pipeline runs independently on each
candidate, and the pairwise verdict is derived from the two ZTR scores.

Determinism / resumability
--------------------------
Shards are independent. A shard that already has a `.done` marker is skipped, so
a requeued array task resumes rather than redoes. Raw generations for every agent
call are written out -- the contamination analysis in Phase 3 reads Proposer text
directly, and re-running the grid to recover it would be expensive.

Local datasets
--------------
`--local-jsonl` reads items from a file instead of the Hub. Records must carry
the CodeJudgeBench field names, so `build_query` and `build_candidates` work
unchanged; `--dataset` and `--split` then act as labels that determine the
output cell name. Used for the HumanEvalFix comparison, where the same harness
must run on a benchmark with hand-written bugs rather than LLM-generated ones.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import march_verbatim as MV
import factscore_code as FS
import cove_code as CV
import march_code as MC


# ---------------------------------------------------------------------------
# CodeJudgeBench verbatim judge prompt + parsing (condition C0)
# ---------------------------------------------------------------------------
# Copied from CodeJudgeBench src/models/pairwise.py PairwiseModel.JUDGE_PROMPT
JUDGE_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is better: [[A>B]] 
2. Assistant B is better: [[B>A]]

Example output: "My final verdict is Assistant A is better: [[A>B]]".

<|User Prompt|>
{query}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>
""".strip()

RESPONSE_PATTERNS_A = ["[[A>B]]", "boxed{A>B}"]
RESPONSE_PATTERNS_B = ["[[B>A]]", "boxed{B>A}"]


def extract_pairwise_pred(text, finish_reason):
    """Verbatim port of PairwiseModel.extract_response."""
    if finish_reason != "stop":
        return ""
    if "</think>" in text:
        text = text.split("</think>")[1]
    pred_a = max([text.rfind(p) for p in RESPONSE_PATTERNS_A])
    pred_b = max([text.rfind(p) for p in RESPONSE_PATTERNS_B])
    if pred_a > pred_b:
        return "A"
    if pred_a < pred_b:
        return "B"
    return ""


# ---------------------------------------------------------------------------
# Task query construction -- ports CodeJudgeBench src/tasks/*.py
# ---------------------------------------------------------------------------
CODEREPAIR_TASK_PROMPT = """
You are provided with a coding question, a piece of code, and the associated error message. 
You need to generate a corrected version of the code.

### Question
{question_content}

### Program to be fixed:
```python
{wrong_code}
```


### Error Information:
{error_message}
""".strip()


def coderepair_error_message(metadata):
    """Verbatim port of CodeRepairTask._get_error_message."""
    ec = metadata["error_code"]
    if ec == -1:
        return f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
    elif ec == -2:
        return (f"The above code is incorrect and got a wrong answer.\nInput:\n{metadata['inputs']}\n\n"
                f"Generated Output:\n{metadata['output']}\n\nExpected:\n{metadata['expected']}")
    elif ec == -3:
        return (f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\n"
                f"Input:\n{metadata['inputs']}\n\nExpected:\n{metadata['expected']}")
    elif ec == -4:
        if "inputs" not in metadata:
            return f"The above code is incorrect and got a runtime error.\n{metadata['error_message']}"
        return (f"The above code is incorrect and got a runtime error.\nInput:\n{metadata['inputs']}\n\n"
                f"Expected:\n{metadata['expected']}\n\n{metadata['error']}")
    raise NotImplementedError(f"error_code={ec} not implemented")


def build_query(item, task):
    """The <|User Prompt|> content for an item, matching CodeJudgeBench."""
    if task == "codegen":
        return item["question_content"]
    if task == "coderepair":
        return CODEREPAIR_TASK_PROMPT.format(
            question_content=item["question_content"],
            wrong_code=item["wrong_code"],
            error_message=coderepair_error_message(item["wrong_meta"]),
        )
    if task == "testgen":
        return ("For the below coding question, give a unit test including test input and test output.\n"
                f"### Question\n{item['question_content']}")
    raise ValueError(task)


def build_candidates(item, task):
    """(positive_text, negative_text) as they appear to the judge."""
    if task == "testgen":
        ut = "### Test Input\n```\n{test_in}\n```\n\n### Test Output\n```\n{answer}\n```"
        return (
            ut.format(test_in=item["pos_input"], answer=item["pos_response"]),
            ut.format(test_in=item["neg_input"], answer=item["neg_response"]),
        )
    return item["pos_response"], item["neg_response"]


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------
def make_llm(args):
    from vllm import LLM
    n_gpu = len((os.getenv("CUDA_VISIBLE_DEVICES") or "0").split(","))
    return LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=n_gpu,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
    )


def make_sampling(args, n=1):
    """CodeJudgeBench Qwen3 params, verbatim (src/models/qwen3.py)."""
    from vllm import SamplingParams
    return SamplingParams(
        n=n,
        max_tokens=args.max_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        seed=args.seed,
    )


class Chat:
    def __init__(self, llm, tokenizer, sampling):
        self.llm, self.tok, self.sp = llm, tokenizer, sampling

    def __call__(self, prompts):
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        outs = self.llm.generate(texts, self.sp)
        return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs]


def strip_think(text):
    return text.split("</think>")[-1] if "</think>" in text else text


# ---------------------------------------------------------------------------
# Condition C0
# ---------------------------------------------------------------------------
def run_c0(items, task, chat):
    recs = []
    for reverse in (False, True):
        prompts = []
        for it in items:
            q = build_query(it, task)
            pos, neg = build_candidates(it, task)
            a, b = (neg, pos) if reverse else (pos, neg)
            prompts.append(JUDGE_PROMPT.format(query=q, answer_a=a, answer_b=b))
        outs = chat(prompts)
        for it, o in zip(items, outs):
            text, fr = o[0]
            recs.append({
                "uid": it["_uid"], "reverse": reverse,
                "label": "B" if reverse else "A",
                "difficulty": it.get("difficulty"),
                "pred": extract_pairwise_pred(text, fr),
                "judge_raw": text,
            })
    return recs


# ---------------------------------------------------------------------------
# Pointwise MARCH pipeline (C1-C4)
# ---------------------------------------------------------------------------
def run_march(items, task, chat, condition, args, traces_map=None):
    verbatim = (condition == "c1_verbatim")

    # ---- Stage 1: Solver, once per (item, side) ----
    units = []
    for it in items:
        q = build_query(it, task)
        pos, neg = build_candidates(it, task)
        for side, cand in (("pos", pos), ("neg", neg)):
            units.append({"uid": it["_uid"], "difficulty": it.get("difficulty"),
                          "side": side, "query": q, "candidate": cand})

    # FActScore has no Solver: its claim generator reads the candidate
    # directly, so the assessment stage is skipped entirely.
    if condition == "c6_factscore":
        for u in units:
            u["solver_raw"] = ""
            u["solver"] = ""
    else:
        solver_out = chat([MC.solver_prompt.format(query=u["query"], candidate=u["candidate"])
                       for u in units])
        for u, o in zip(units, solver_out):
            u["solver_raw"] = o[0][0]
            u["solver"] = strip_think(o[0][0]).strip()

    # ---- Stage 2: Proposer ----
    if verbatim:
        # MARCH's own prompt, unmodified. Its {query}/{response} slots map onto
        # the coding problem and the Solver's analysis.
        pp = [MV.question_prompt.format(query=u["query"], response=u["solver"]) for u in units]
    elif condition == "c6_factscore":
        pp = [FS.claim_prompt.format(query=u["query"],
              candidate=u["candidate"]) for u in units]
    elif condition == "c5_cove":
        pp = [CV.proposer_prompt.format(query=u["query"], response=u["solver"]) for u in units]
    else:
        pp = [MC.proposer_prompt.format(query=u["query"], response=u["solver"]) for u in units]

    prop_out = chat(pp)
    for u, o in zip(units, prop_out):
        u["proposer_raw"] = o[0][0]
        txt = strip_think(o[0][0])
        # Both branches use MARCH's verbatim extractor: the regex is
        # answer-agnostic, so C2's verdict tokens parse with the same code.
        # CoVe questions carry no expected answer and need their own
        # extractor; MARCH's regex requires the [Answer: ...] bracket.
        if condition == "c6_factscore":
            u["qa_pairs"] = FS.extract_claims(txt)
        else:
            u["qa_pairs"] = (CV.extract_qa_pairs(txt) if condition == "c5_cove"
                         else MV.extract_qa_pairs(txt))
        u["n_claims"] = len(u["qa_pairs"])
        u["is_padding"] = (u["n_claims"] == 0)   # MARCH's no-QA-pairs path

    # ---- Stage 3: Checker, 3 samples, majority vote (MARCH majority_voting_nums=3) ----
    live = [u for u in units if not u["is_padding"]]
    if condition == "c5_cove":
        live = []   # decomposition-only condition; no verifier is run
    if live and condition == "c6_factscore":
        # FActScore adjudicates each claim against an external knowledge
        # source and scores by the supported fraction, not zero tolerance.
        cp = []
        for u in live:
            st = "\n".join(f"{i+1}. {qa['question']}"
                            for i, qa in enumerate(u["qa_pairs"]))
            docs = FS.build_docs(u["query"], u["candidate"])
            cp.append(FS.verifier_prompt.format(docs=docs, statements=st))
        chk_out = chat(cp)
        for u, o in zip(live, chk_out):
            n_q = u["n_claims"]
            votes = [0] * n_q
            u["checker_raw"] = []
            for k in range(min(3, len(o))):
                text = strip_think(o[k][0])
                sup = FS.extract_support(text, n_q)
                for qi in range(n_q):
                    votes[qi] += 1 if sup[qi] else -1
                u["checker_raw"].append(o[k][0])
            n_sup = sum(1 for v in votes if v > 0)
            u["true_num"] = n_sup
            u["fault_num"] = n_q - n_sup
            u["total_num"] = n_q
            u["score"] = FS.supported_fraction(n_sup, n_q)
        live = []          # scored above; skip the MARCH verifier below

    if live:
        cp = []
        for u in live:
            qs = "\n".join(f"- Question {i+1}:{qa['question']}"
                           for i, qa in enumerate(u["qa_pairs"]))
            if verbatim:
                docs = MC.build_docs_static(u["query"], u["candidate"])
                prompt = MV.answer_question_prompt.format(docs=docs, questions=qs)
            elif condition == "c4_exec":
                tr = (traces_map or {}).get((u["uid"], u["side"]), [])
                docs = MC.build_docs_execution(u["query"], u["candidate"], tr)
                prompt = MC.checker_prompt.format(docs=docs, questions=qs)
            else:
                docs = MC.build_docs_static(u["query"], u["candidate"])
                prompt = MC.checker_prompt.format(docs=docs, questions=qs)
                if condition == "c3_unblinded":
                    prompt += MC.unblinded_suffix.format(solver_response=u["solver"])
            cp.append(prompt)

        chk_out = chat(cp)  # sampling n=3 -> three trajectories per prompt
        eq = MV.check_ans_eq if verbatim else MC.check_verdict_eq

        for u, o in zip(live, chk_out):
            n_q = u["n_claims"]
            correct_count = [0] * n_q
            u["checker_raw"] = []
            for k in range(min(3, len(o))):
                text = strip_think(o[k][0])
                if not text.strip().startswith("1."):
                    text = "1." + text.strip()
                re_ans = MV.extract_detailed_info(text)
                info = []
                for qi, qa in enumerate(u["qa_pairs"]):
                    if qi + 1 not in re_ans:
                        info.append(-1)
                    elif eq(re_ans[qi + 1]["answer"], qa["answer"]):
                        info.append(1)
                    else:
                        info.append(-1)
                for qi in range(n_q):
                    correct_count[qi] += info[qi]
                u["checker_raw"].append(o[k][0])
                u[f"correct_info_{k}"] = info

            u["fault_num"] = sum(1 for c in correct_count if c < 0)
            u["true_num"] = sum(1 for c in correct_count if c > 0)
            u["total_num"] = n_q
            u["score"] = MV.ztr_score(u["fault_num"], n_q, use_ztr=args.use_ztr)

    for u in units:
        if u["is_padding"]:
            u.update(fault_num=0, true_num=0, total_num=0, score=0.0, checker_raw=[])

    # c5_cove runs no verifier, so no unit carries a score. That
    # condition compares claim sets rather than verdicts, so both sides
    # get 0.0 and every pair reads as a tie.
    for u in units:
        u.setdefault("score", 0.0)
        u.setdefault("fault_num", 0)
        u.setdefault("true_num", 0)
        u.setdefault("total_num", 0)
        u.setdefault("checker_raw", [])

    # ---- Stage 4: pointwise -> pairwise ----
    by_uid = {}
    for u in units:
        by_uid.setdefault(u["uid"], {})[u["side"]] = u

    recs = []
    for uid, sides in by_uid.items():
        p, n = sides.get("pos"), sides.get("neg")
        if p is None or n is None:
            continue
        # Higher (less negative) ZTR score => judged more likely correct.
        if p["score"] > n["score"]:
            verdict, tie = "pos", False
        elif p["score"] < n["score"]:
            verdict, tie = "neg", False
        else:
            verdict, tie = "tie", True

        for reverse in (False, True):
            label = "B" if reverse else "A"
            if tie:
                pred = ""
            elif verdict == "pos":
                pred = "B" if reverse else "A"
            else:
                pred = "A" if reverse else "B"
            recs.append({
                "uid": uid, "reverse": reverse, "label": label,
                "difficulty": p["difficulty"], "pred": pred, "tie": tie,
                "pos_score": p["score"], "neg_score": n["score"],
                "pos_n_claims": p["n_claims"], "neg_n_claims": n["n_claims"],
                "pos_padding": p["is_padding"], "neg_padding": n["is_padding"],
                "pos_fault": p["fault_num"], "neg_fault": n["fault_num"],
                "pos_qa_pairs": p["qa_pairs"], "neg_qa_pairs": n["qa_pairs"],
                "pos_solver": p["solver_raw"], "neg_solver": n["solver_raw"],
                "pos_proposer": p["proposer_raw"], "neg_proposer": n["proposer_raw"],
                "pos_checker": p["checker_raw"], "neg_checker": n["checker_raw"],
            })
    return recs


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["c0_direct", "c1_verbatim", "c2_adapted",
                             "c3_unblinded", "c4_exec", "c5_cove", "c6_factscore"])
    # PATCH 1: humanevalfix added. --dataset feeds the output cell name in
    # xfer_cell.sh, so a local benchmark needs its own value rather than
    # reusing an existing one.
    ap.add_argument("--dataset", default="codejudgebench",
                    choices=["codejudgebench", "codejudgebench-adversarial",
                             "humanevalfix"])
    ap.add_argument("--task", required=True,
                    choices=["codegen", "coderepair", "testgen"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap items")
    # PATCH 2: local JSONL source.
    ap.add_argument("--local-jsonl", default=None,
                    help="read items from a local JSONL instead of the Hub. "
                         "Records must carry the CodeJudgeBench field names "
                         "(question_content, pos_response, neg_response, "
                         "difficulty); --dataset/--split then act as labels "
                         "only, and determine the output cell name.")
    ap.add_argument("--difficulty", default="all",
                    choices=["all", "easy", "medium", "hard", "medium+hard"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", default="", help="C4: JSONL of execution traces")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--use-ztr", type=int, default=1)
    args = ap.parse_args()
    args.use_ztr = bool(args.use_ztr)

    done = args.out + ".done"
    if os.path.exists(done):
        print(f"[skip] {done} exists", flush=True)
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # PATCH 3: branch the load. The _uid loop stays outside the branch -- both
    # paths need it, and keeping the {task}-{split}-{i} format means uids stay
    # comparable and the .done resume logic is unaffected.
    if args.local_jsonl:
        with open(args.local_jsonl) as _fh:
            ds = [json.loads(_l) for _l in _fh if _l.strip()]
        print(f"[info] local jsonl: {len(ds)} items from {args.local_jsonl}",
              flush=True)
    else:
        from datasets import load_dataset
        ds = load_dataset(f"mattymchen/{args.dataset}", args.task)[args.split].to_list()
    for i, it in enumerate(ds):
        it["_uid"] = f"{args.task}-{args.split}-{i}"

    if args.difficulty != "all":
        keep = {"medium", "hard"} if args.difficulty == "medium+hard" else {args.difficulty}
        ds = [d for d in ds if d.get("difficulty") in keep]
    if args.limit:
        ds = ds[: args.limit]
    items = ds[args.shard::args.nshards]
    print(f"[info] {args.condition} {args.task}/{args.split} "
          f"shard {args.shard}/{args.nshards}: {len(items)} items", flush=True)
    if not items:
        open(done, "w").close()
        return

    traces_map = None
    if args.condition == "c4_exec":
        if not args.traces or not os.path.exists(args.traces):
            sys.exit("[fatal] --traces required for c4_exec")
        traces_map = {}
        with open(args.traces) as f:
            for line in f:
                r = json.loads(line)
                traces_map[(r["uid"], r["side"])] = r["traces"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = make_llm(args)

    t0 = time.time()
    if args.condition == "c0_direct":
        chat = Chat(llm, tok, make_sampling(args, n=1))
        recs = run_c0(items, args.task, chat)
    else:
        chat1 = Chat(llm, tok, make_sampling(args, n=1))
        chat3 = Chat(llm, tok, make_sampling(args, n=3))

        class Dual:
            """Solver/Proposer use n=1; Checker uses n=3 (MARCH majority voting)."""
            def __init__(self): self.calls = 0
            def __call__(self, prompts):
                self.calls += 1
                return (chat3 if self.calls == 3 else chat1)(prompts)

        recs = run_march(items, args.task, Dual(), args.condition, args, traces_map)

    with open(args.out, "w") as f:
        for r in recs:
            r["condition"] = args.condition
            r["dataset"] = args.dataset
            r["task"] = args.task
            r["split"] = args.split
            f.write(json.dumps(r) + "\n")
    open(done, "w").close()
    print(f"[done] {len(recs)} rows -> {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
