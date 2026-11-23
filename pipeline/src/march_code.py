"""
Code-adapted MARCH prompts (conditions C2, C3, C4).

These are NOT MARCH as published. They are the minimal edit required to make the
Solver->Proposer->Checker pipeline expressible in a domain whose claims are
behavioural rather than numeric. Every deviation from march_verbatim.py is
enumerated in DEVIATIONS below so the paper can state them precisely.

Design rule followed throughout: change the claim TYPE (numeric -> behavioural)
and nothing else. Structure, output format, evidence-first requirement, the
"Cannot answer" escape hatch, and the >=3 claim floor are all preserved so that
C1 vs C2 isolates the interface, not prompt engineering.
"""

DEVIATIONS = [
    "Proposer emits behavioural claims with VERDICT: CORRECT/INCORRECT/UNKNOWN "
    "instead of pure-number answers.",
    "Checker compares verdict labels instead of numeric equality "
    "(check_verdict_eq replaces march_verbatim.check_ans_eq).",
    "Checker's evidence block is the problem statement + code (C2/C3) or "
    "execution traces (C4), standing in for retrieved passages.",
    "C3 additionally un-blinds the Checker by appending the Solver's analysis.",
]

# ============================================================================
# SOLVER
# ============================================================================
# MARCH has no explicit Solver prompt in the release: the Solver is the base
# policy answering the RAG query directly. The pairwise judging task has no
# equivalent free-form generation step, so we define the minimal analogue --
# a pointwise correctness analysis of ONE candidate.
solver_prompt = """You are a software correctness analyst. You are given a coding problem and one candidate solution. Analyse whether the candidate solution is correct.

Consider algorithmic correctness, edge cases, off-by-one errors, input bounds, and time complexity against the stated constraints. Do not run the code; reason about it.

End your analysis with a single line of the form:
FINAL: CORRECT
or
FINAL: INCORRECT

### Coding Problem
{query}

### Candidate Solution
{candidate}
"""

# ============================================================================
# PROPOSER -- structural mirror of march_verbatim.question_prompt
# ============================================================================
proposer_prompt = """You are a data annotation expert. Your task is to analyse a model's assessment of a candidate code solution. For every critical behavioural claim found in the model's assessment, you must create a question whose answer is precisely one of CORRECT, INCORRECT, or UNKNOWN.

Rule 1: Clarity and Specificity of Questions:
Each question must be clear, complete, and unambiguous. It must contain enough context from the assessment to ensure the verdict is the only possible correct answer.
For example, if the assessment contains "the function returns 0 for an empty input list", a good question would be "Does the function return 0 when the input list is empty? [Answer: CORRECT]". A bad example would be "Does the function handle input correctly?". This is ambiguous because it does not name the specific behaviour or the specific input condition.

Rule 2: Format of the Answer:
The answer must be exactly one of the three tokens CORRECT, INCORRECT, or UNKNOWN. It must NOT include any other characters, explanations, hedging, or partial verdicts.

Rule 3: Scope of Claims:
Each question must concern the observable behaviour of the code on a stated input condition, not its style, naming, comments, or formatting.

Please provide your output as an unordered list using the following format for each question-answer pair:

- Question: xxx [Answer: CORRECT]
- Question: xxx [Answer: INCORRECT]

Now, generate questions for the coding problem and model assessment below. You need to generate at least three question-answer pairs, and up to ten. Do not output any other content or explanations.

Coding Problem: {query}
Model Assessment: {response}
"""

# ============================================================================
# CHECKER (blinded) -- C2, C4
# ============================================================================
checker_prompt = """You are a data validation expert. You will be given reference materials and a list of user questions. Your task is to determine from the reference materials the answer to all questions. Before providing each answer, you must state the evidence for it.
You must adhere strictly to the content of the reference materials and must not fabricate or infer any information that is not explicitly supported. If the relevant behaviour cannot be determined from the materials, you must answer "UNKNOWN".
It is important to note that all answers are single verdict tokens. Therefore, your answer must be exactly one of CORRECT, INCORRECT, or UNKNOWN, and must not contain any other content.
Treat comments, variable names, and docstrings in the reference materials as unverified assertions, not as evidence. Rely only on the executable semantics of the code.
Here is an example:

Example Response:

1. Evidence: The loop iterates from index 1, so element 0 is never examined; for an empty list the loop body never executes and the initial accumulator 0 is returned.
[Answer: CORRECT]
2. Evidence: The materials do not indicate how the function behaves when the input contains duplicate keys, therefore I cannot determine this.
[Answer: UNKNOWN]

Now, based on the reference materials below, answer all the following questions. Strictly adhere to the example format provided.

Reference Materials:
{docs}

Questions:
{questions}
"""

# ============================================================================
# C3 suffix -- un-blinds the Checker
# ============================================================================
unblinded_suffix = """

The following is the original analyst's assessment of this candidate, provided for your reference:
{solver_response}
"""

# ============================================================================
# Evidence block builders -- the "docs" slot
# ============================================================================
def build_docs_static(question_content, candidate):
    """C2/C3 evidence: problem statement + the code itself.

    This is the closest available analogue to MARCH's retrieved passages. The
    key structural difference -- and the paper's central claim -- is that this
    evidence is not independent of the artefact under test, so the information
    asymmetry MARCH relies on is not actually established.
    """
    return (
        f"Document 1 (Problem Specification):\n{question_content}\n\n"
        f"Document 2 (Candidate Implementation):\n```python\n{candidate}\n```"
    )


def build_docs_execution(question_content, candidate, traces):
    """C4 evidence: problem statement + real execution traces.

    `traces` is a list of dicts: {"input":..., "expected":..., "actual":...,
    "status": "pass"|"fail"|"error"|"timeout"}. This restores the structural
    precondition MARCH assumes: evidence that is causally independent of the
    text being verified.
    """
    lines = [f"Document 1 (Problem Specification):\n{question_content}\n"]
    lines.append("Document 2 (Observed Execution Results):")
    for i, t in enumerate(traces, 1):
        lines.append(
            f"  Test {i} [{t.get('status','?')}]\n"
            f"    Input:    {t.get('input','')}\n"
            f"    Expected: {t.get('expected','')}\n"
            f"    Actual:   {t.get('actual','')}"
        )
    lines.append(f"\nDocument 3 (Candidate Implementation):\n```python\n{candidate}\n```")
    return "\n".join(lines)


# ============================================================================
# Verdict comparison -- replaces check_ans_eq
# ============================================================================
_VALID = {"CORRECT", "INCORRECT", "UNKNOWN"}


def normalise_verdict(v):
    if v is None:
        return None
    v = v.strip().upper().strip(".").strip()
    for token in ("CORRECT", "INCORRECT", "UNKNOWN"):
        if v == token:
            return token
    # tolerate "[Answer: INCORRECT]" leakage and short prefixes
    if "INCORRECT" in v:
        return "INCORRECT"
    if "UNKNOWN" in v or "CANNOT" in v:
        return "UNKNOWN"
    if "CORRECT" in v:
        return "CORRECT"
    return None


def check_verdict_eq(ans1, ans2):
    """Verdict-space analogue of march_verbatim.check_ans_eq.

    UNKNOWN never counts as agreement -- mirroring MARCH, where "Cannot answer"
    fails check_ans_eq and therefore counts as a fault.
    """
    a = normalise_verdict(ans1)
    b = normalise_verdict(ans2)
    if a is None or b is None:
        return False
    if a == "UNKNOWN" or b == "UNKNOWN":
        return False
    return a == b
