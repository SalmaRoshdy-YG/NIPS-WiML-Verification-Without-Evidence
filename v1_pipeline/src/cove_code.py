"""
cove_code.py -- CoVe's verification-planning structure, adapted to code.

Rules 1 and 2 are CoVe's Factored variant: questions must be specific and
independently answerable without the baseline in view. Rule 3 is the deliberate
contrast with MARCH, whose Proposer requires "the answer must be a pure number".
Everything else in the pipeline is held fixed, so the answer constraint is the
only variable between c1_verbatim and c5_cove.

This is an adaptation of CoVe's decomposition structure to code, not a port of
an implementation: no official CoVe implementation exists. See RQ2.
"""
import re

proposer_prompt = """You are a verification planner. You are given a coding problem and an analyst's assessment of one candidate solution. Plan a set of verification questions that would establish whether the assessment is correct.

Rule 1: Each question must be clear, complete, and unambiguous. It must contain enough context from the assessment that the answer is determined.

Rule 2: Each question must be answerable on its own, without reference to the other questions and without seeing the assessment again.

Rule 3: Place no constraint on the form of the answer. A question may ask about any property of the solution that could be established or refuted.

Please provide your output as an unordered list using the following format:

- Question: xxx
- Question: xxx

Now, generate questions for the coding problem and assessment below. You need to generate at least three questions, and up to ten. Do not output any other content or explanations.

Coding Problem: {query}
Model Assessment: {response}
"""

_Q = re.compile(r"^\s*-?\s*Question\s*\d*\s*:\s*(.+?)\s*$", re.M)


def extract_qa_pairs(text):
    """CoVe questions carry no expected answer, so `answer` is left empty.

    Returns the same shape as MARCH's extractor so downstream code and the
    analysis scripts need no changes.
    """
    out, seen = [], set()
    for q in _Q.findall(text or ""):
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append({"question": q, "answer": ""})
    return out


# ============================================================================
# c8_cove -- the full Factored CoVe pipeline: baseline, plan, execute, revise.
#
# The baseline is the Solver's assessment (it already ends in a FINAL line),
# planning is proposer_prompt above, and the two stages below complete it.
# Execution is factored: each question is answered in its own prompt, against
# the same evidence MARCH's verifier sees, and without the assessment in view.
# ============================================================================
answer_prompt = """You are given a coding problem and one candidate solution. Answer the question below about the candidate solution. Reason from the problem and the code; do not run the code.

Answer concisely, in at most three sentences, and state the evidence for your answer.

{docs}

Question: {question}
"""

final_prompt = """You are a software correctness analyst. You are given a coding problem, one candidate solution, an initial assessment of whether the solution is correct, and a set of verification questions with answers obtained independently of that assessment.

Using the verification answers, correct any mistakes in the initial assessment and decide whether the candidate solution is correct. Where the verification answers conflict with the initial assessment, prefer the verification answers.

End your response with a single line of the form:
FINAL: CORRECT
or
FINAL: INCORRECT

### Coding Problem
{query}

### Candidate Solution
```python
{candidate}
```

### Initial Assessment
{response}

### Verification Questions and Answers
{qa}
"""

_FINAL = re.compile(r"FINAL:\s*\**\s*(CORRECT|INCORRECT)\b", re.I)


def extract_final(text):
    """Last FINAL verdict in the text, upper-cased, or None if absent."""
    m = _FINAL.findall(text or "")
    return m[-1].upper() if m else None


def final_score(verdict):
    """CORRECT 1.0, INCORRECT 0.0; an unparsed verdict sits at 0.5, so two
    unparsed candidates tie and one is never ranked as a definite failure."""
    return {"CORRECT": 1.0, "INCORRECT": 0.0}.get(verdict, 0.5)
