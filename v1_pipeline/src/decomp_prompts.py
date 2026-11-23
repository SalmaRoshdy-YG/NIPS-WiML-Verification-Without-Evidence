"""
decomp_prompts.py -- the decomposition prompt for each framework compared in RQ2.

Every external entry must be filled from that framework's own source before
use. Do not paraphrase from memory: the point of the comparison is that the
decomposition instruction belongs to the framework and not to us, and a
paraphrase turns the result into a claim about our wording. decomp_probe.py
refuses to run any framework whose `text` still contains FILL_ME.

Substitution fields are {problem} and {solution}. Rename each framework's own
fields to these and record the renaming under `deviations`.

`decomposes` says what the framework decomposes, and it changes what the probe
must run:

    "candidate"   the artifact is decomposed directly, so {solution} is the
                  candidate code. FActScore and VeriScore work this way.
    "assessment"  the framework first writes a prose assessment of the artifact
                  and decomposes that, so the probe runs SOLVER first and
                  {solution} is the assessment. MARCH and CoVe work this way.

Getting this wrong silently misrepresents a framework. Feeding MARCH's Proposer
the raw candidate would give it an input it never receives in the pipeline, and
the measured rate would not be MARCH's.

Where to source the external prompts
------------------------------------
factscore  Min et al. 2023, arXiv:2305.14251, atomic-fact decomposition prompt.
           Reused verbatim by later work. Repo: github.com/shmsw25/FActScore
cove       Dhuliawala et al. 2023, arXiv:2309.11495, plan-verifications stage,
           Factored variant, whose questions are answered independently and so
           is the analogue of MARCH's blinded Checker. There is NO official
           implementation; the repo that appears in search results is a
           third-party reimplementation and is not canonical. Use the paper's
           appendix template and say so.
veriscore  Song et al. 2024, Findings of EMNLP.
           Repo: github.com/Yixiao-Song/VeriScore
           Claim extraction runs standalone:
               python3 -m veriscore.extract_claims --data_dir D \\
                   --input_file F --model_name M --use_external_model
           Input is JSONL with a `response` field; `question` is optional and
           switches it to QA prompting. Prefer running their extractor over
           copying the prompt, and pass a local model so the comparison does
           not confound framework with model.
"""

FILL = "FILL_ME"

# Run before the Proposer for any framework whose `decomposes` is "assessment".
# Ported unmodified from march_code.py line 32.
SOLVER = """You are a software correctness analyst. You are given a coding problem and one candidate solution. Analyse whether the candidate solution is correct.

Consider algorithmic correctness, edge cases, off-by-one errors, input bounds, and time complexity against the stated constraints. Do not run the code; reason about it.

End your analysis with a single line of the form:
FINAL: CORRECT
or
FINAL: INCORRECT

### Coding Problem
{problem}

### Candidate Solution
{candidate}
"""

PROMPTS = {
    # ------------------------------------------------------------------ MARCH
    "march_c1": {
        "text": """You are a data annotation expert. Your task is to analyze a model's response to a user query. For every critical number found in the model's response, you must create a question where the answer is precisely and exactly that number.

Rule 1: Clarity and Specificity of Questions:
Each question must be clear, complete, and unambiguous. It must contain enough context from the response to ensure the number is the only possible correct answer.
For example, if the model's response contains "In 2025, 50 people will take the bar exam in Beijing," a good question would be "How many people will take the bar exam in the Beijing area in 2025?", and the answer is 50.
A bad example would be "How many people will take the bar exam in 2025?". This is ambiguous because it lacks the specific region ('Beijing area').

Rule 2: Format of the Answer:
The answer must be a pure number, either an integer or a decimal. It must NOT include any other characters or formats, such as percentage signs (%), 'k' to denote thousands (e.g., 10k), ranges (e.g., 10-20), or words (e.g., fifty).

Please provide your output as an unordered list using the following format for each question-answer pair:

- Question: xxx [Answer: n]
- Question: xxx [Answer: n]

Now, generate questions for the user query and model response below. You need to generate at least three question-answer pairs, and up to ten. Do not output any other content or explanations.

User Query: {problem}
Model Response: {solution}
""",
        "source": "MARCH release, march_verbatim.py line 20, ported unmodified",
        "deviations": "fields {query}/{response} renamed to {problem}/{solution}",
        "answer_space": "numeric",
        "decomposes": "assessment",
    },

    "march_c2": {
        "text": """You are a data annotation expert. Your task is to analyse a model's assessment of a candidate code solution. For every critical behavioural claim found in the model's assessment, you must create a question whose answer is precisely one of CORRECT, INCORRECT, or UNKNOWN.

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

Coding Problem: {problem}
Model Assessment: {solution}
""",
        "source": "our adaptation of the above; see Appendix B",
        "deviations": "answer vocabulary replaced with the three-token verdict; "
                      "fields renamed to {problem}/{solution}",
        "answer_space": "three-token verdict",
        "decomposes": "assessment",
    },

    # -------------------------------------------------------------- FActScore
    "factscore": {
        "text": FILL,
        "source": "Min et al. 2023, atomic-fact decomposition prompt, verbatim",
        "deviations": FILL,
        "answer_space": "unconstrained; items are statements, not questions",
        "decomposes": "candidate",
    },

    # ------------------------------------------------------------------- CoVe
    "cove": {
        "text": FILL,
        "source": "Dhuliawala et al. 2023, appendix, plan-verifications "
                  "template, Factored variant; no official implementation",
        "deviations": FILL,
        "answer_space": "unconstrained",
        "decomposes": "assessment",
    },

    # -------------------------------------------------------------- VeriScore
    # Fallback only. Prefer running their extractor; record which route you took.
    "veriscore": {
        "text": FILL,
        "source": "Song et al. 2024; github.com/Yixiao-Song/VeriScore",
        "deviations": FILL,
        "answer_space": "unconstrained, filtered to verifiable claims only",
        "decomposes": "candidate",
    },
}


def get(name: str) -> dict:
    if name not in PROMPTS:
        raise KeyError(f"unknown framework {name!r}; have {sorted(PROMPTS)}")
    p = PROMPTS[name]
    if p.get("decomposes") not in ("candidate", "assessment"):
        raise SystemExit(f"{name!r}: 'decomposes' must be 'candidate' "
                         f"or 'assessment'")
    if FILL in (p["text"] or ""):
        raise SystemExit(
            f"\nThe prompt for {name!r} has not been filled in.\n"
            f"Source it from: {p['source']}\n"
            f"See the module docstring for where to look.\n"
            f"Refusing to run rather than measure our own paraphrase.\n")
    return p
