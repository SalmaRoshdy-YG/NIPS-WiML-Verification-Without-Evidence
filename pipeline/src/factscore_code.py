"""
factscore_code.py -- FActScore's decompose-and-verify shape, applied to code.

FActScore \\citep{factscore} is the oldest member of the lineage this paper
studies and the simplest thing that could exhibit claim invariance. It has no
Solver, no blinding and no agent topology: a response is decomposed into atomic
facts, each fact is checked against an external knowledge source, and the
response is scored by the fraction of its facts that the source supports.

Three consequences of that shape, each of which differs from MARCH:

  1. No Solver. The claim generator reads the candidate directly, so there is
     no intermediate assessment to decompose.
  2. Evidence is external. FActScore checks facts against a knowledge source
     that is not the response. Here the only external source available is the
     problem statement, so DOCS_MODE defaults to "spec". That makes the
     evidence identical for both candidates by construction, which is the
     condition of Section 3 at its starkest.
  3. Scoring is the supported fraction, not zero tolerance. A single
     unsupported claim does not floor a candidate.

We use FActScore's bare decomposition instruction. The released implementation
retrieves few-shot demonstrations by BM25 similarity from a corpus of biography
sentences, which would supply nothing useful here; we therefore run it without
demonstrations and record that in the deviations table rather than substituting
code examples of our own, which would make the measurement ours rather than
theirs.
"""
import re

# Set to "spec+code" to give the verifier the candidate as well. That is not
# FActScore's shape -- its knowledge source is external to the response -- but
# it makes the evidence comparable to C2's and is reported as a variant.
DOCS_MODE = "spec"


# ============================================================================
# CLAIM GENERATOR -- FActScore's instruction, no demonstrations
# ============================================================================
claim_prompt = """Please breakdown the following candidate solution into independent facts. Each fact must be a single short statement about what the code does, must stand on its own, and must be checkable without seeing the other facts.

Write one fact per line, in the form:

- <fact>
- <fact>

Write between three and ten facts. Do not output any other content or explanations.

### Problem
{query}

### Candidate Solution
{candidate}
"""


# ============================================================================
# VERIFIER -- adjudicates each claim against the knowledge source
# ============================================================================
verifier_prompt = """You are a data validation expert. You will be given reference materials and a list of statements. Your task is to determine, for each statement, whether the reference materials support it. Before providing each answer, you must state the evidence for it.
You must adhere strictly to the content of the reference materials and must not fabricate or infer any information that is not explicitly supported. If the materials do not settle the statement, answer NOT SUPPORTED.
It is important to note that all answers are single verdict tokens. Therefore, your answer must be exactly one of SUPPORTED or NOT SUPPORTED, and must not contain any other content.
Here is an example:

Example Response:

1. Evidence: The specification states that the function must return 0 for an empty input, and the statement asserts exactly that.
[Answer: SUPPORTED]
2. Evidence: The materials say nothing about behaviour on duplicate keys, so the statement cannot be settled.
[Answer: NOT SUPPORTED]

Now, based on the reference materials below, answer for all the following statements. Strictly adhere to the example format provided.

Reference Materials:
{docs}

Statements:
{statements}
"""


def build_docs(question_content, candidate, mode=None):
    """FActScore's knowledge source.

    In "spec" mode the source is the problem statement alone, which is what
    makes it external to the response and identical across the two candidates.
    """
    if (mode or DOCS_MODE) == "spec+code":
        return (f"Document 1 (Problem Specification):\n{question_content}\n\n"
                f"Document 2 (Candidate Implementation):\n"
                f"```python\n{candidate}\n```")
    return f"Document 1 (Problem Specification):\n{question_content}"


_CLAIM = re.compile(r"^\s*[-*\u2022]?\s*(?:\d+[.)]\s*)?(.+?)\s*$", re.M)


def extract_claims(text):
    """Parse the declarative claim list.

    Returns MARCH's qa_pairs shape so the surrounding pipeline and the
    analysis scripts need no changes. FActScore claims carry no expected
    answer, so `answer` is empty and agreement is not defined for them; the
    score is the supported fraction instead.
    """
    out, seen = [], set()
    for line in (text or "").splitlines():
        m = _CLAIM.match(line)
        if not m:
            continue
        c = " ".join(m.group(1).split())
        # drop headings, blanks and anything too short to be a claim
        if len(c.split()) < 4 or c.endswith(":"):
            continue
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append({"question": c, "answer": ""})
    return out[:10]


_SUP = re.compile(r"\[Answer:\s*(NOT\s+SUPPORTED|SUPPORTED)\s*\]", re.I)


def extract_support(text, n_claims):
    """Return a list of booleans, one per claim, in the verifier's order.

    Anything unparsed counts as not supported, which is the conservative
    reading and matches how MARCH treats an unmatched answer.
    """
    found = [bool(re.fullmatch(r"SUPPORTED", m.strip(), re.I))
             for m in _SUP.findall(text or "")]
    found = found[:n_claims]
    return found + [False] * (n_claims - len(found))


def supported_fraction(n_supported, n_claims):
    """FActScore's score: the fraction of claims the source supports.

    A candidate for which no claims were generated scores 0.0, mirroring
    MARCH's no-claims path, so that an unchecked candidate and a fully
    unsupported one are not distinguished. Higher is better, where MARCH's
    zero-tolerance score is 0.0 or -1.0.
    """
    if n_claims <= 0:
        return 0.0
    return n_supported / n_claims
