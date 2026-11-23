"""
MARCH primitives, copied VERBATIM from the official release.

Source: Qwen-Applications/MARCH
  quarl/quarl/trainer/ray_trainer_with_fact_check_march.py

Nothing in this file is modified. Prompts are byte-identical to lines 36-82 of
the source; the three parsing/comparison functions are lifted from the
MARCHFactCheckTrainer methods `_extract_qa_pairs` (L550), `_extract_detailed_info`
(L561) and `_check_ans_eq` (L575), with `self` dropped so they are module-level.

This isolation is deliberate and load-bearing for the paper: the C1 condition
must be defensibly "MARCH as published", so every string and every regex used in
C1 lives here and is never edited. Adaptations belong in march_code.py.
"""

# ============================================================================
# PROPOSER prompt -- verbatim, source L36-52
# ============================================================================
question_prompt = """You are a data annotation expert. Your task is to analyze a model's response to a user query. For every critical number found in the model's response, you must create a question where the answer is precisely and exactly that number.

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

User Query: {query}
Model Response: {response}
"""

# ============================================================================
# CHECKER prompt -- verbatim, source L54-82
# ============================================================================
answer_question_prompt = r"""You are a data validation expert. You will be given a set of reference materials and a list of user questions. Your task is to retrieve information from the reference materials to answer all questions. Before providing each answer, you must state the evidence for it.
You must adhere strictly to the content of the reference materials and must not fabricate or infer any information that is not explicitly mentioned. If the relevant information cannot be found in the materials, you must state "Cannot answer".
It is important to note that all answers are pure numbers. Therefore, your answer must also be a pure number and must not contain any other content, such as percentage signs (%), 'k' to denote thousands, ranges, or numbers written as words.
Here is an example:
Reference Materials:
Document 1: ...in 2024, 50 people will take the bar exam in Beijing. \nDocument 2: ...(omitted)

- Question 1: How many people will take the bar exam in Beijing in 2024?
- Question 2: How many people will take the bar exam in Beijing in 2025?

Example Response:

1. Evidence: Document 1 states that 50 people will take the bar exam in Beijing in 2024.
[Answer: 50]
2. Evidence: The materials do not contain information about the bar exam in Beijing for 2025, therefore I cannot answer.
[Answer: Cannot answer]

Now, based on the reference materials below, answer all the following questions. Strictly adhere to the example format provided.

Reference Materials:
{docs}

Questions:
{questions}
"""

import re


def extract_qa_pairs(text):
    """VERBATIM from MARCHFactCheckTrainer._extract_qa_pairs (source L550)."""
    pattern = r"-\s*Question:\s*(.*?)\s*\[Answer:\s*(.*?)\]"
    matches = re.findall(pattern, text)

    qa_pairs = []
    for question, answer in matches:
        qa_pairs.append({"question": question.strip(), "answer": answer.strip()})

    return qa_pairs


def extract_detailed_info(text):
    """VERBATIM from MARCHFactCheckTrainer._extract_detailed_info (source L561).

    The debug `print(matches)` in the original is dropped; behaviour identical.
    """
    pattern = r"(\d+)\.\s*Evidence:\s*(.*?)\s*\[Answer:\s*(.*?)\]"
    matches = re.findall(pattern, text, re.DOTALL)

    result = {}

    for index, match in enumerate(matches):
        number, basis, answer = match[0], match[-2], match[-1]
        result[int(number)] = {
            "number": int(number),
            "basis": basis.strip(),
            "answer": answer.strip(),
        }

    return result


def check_ans_eq(ans1, ans2):
    """VERBATIM from MARCHFactCheckTrainer._check_ans_eq (source L575).

    Note the domain assumption baked in here: answers are stripped of
    ['%','k','K','w','W',','] and cast to int/float. Any non-numeric claim that
    is not a literal string match returns False. This is the mechanism that
    makes C1 fail on code, and it is why we run it unmodified.
    """
    for char in ['%', 'k', 'K', 'w', 'W', ',']:
        ans1 = ans1.replace(char, "")
        ans2 = ans2.replace(char, "")

    if ans1 == ans2:
        return True

    try:
        num1 = float(ans1) if "." in ans1 else int(ans1)
    except (ValueError, AttributeError):
        return False

    try:
        num2 = float(ans2) if "." in ans2 else int(ans2)
    except (ValueError, AttributeError):
        return False

    if isinstance(num1, int) and isinstance(num2, int):
        return num1 == num2
    elif isinstance(num1, float) and isinstance(num2, float):
        return abs(num1 - num2) < 0.1
    else:
        return abs(float(num1) - float(num2)) < 0.1


# ============================================================================
# Reward, transcribed from the scoring block at source L855-869
# ============================================================================
def ztr_score(fault_num, total_num, use_ztr=True, coef=0.1):
    """Zero-Tolerance Reward / Error-Rate reward, as in the release.

    ZTR:  -1.0 if any claim failed, else 0.0
    ER :  -coef * min(fault_num, 13)

    The no-QA-pairs path in the release sets score 0.0 and marks the sample
    `is_padding` (source L752-764). We reproduce that in the pipeline, and it is
    the failure mode we expect to dominate C1 on code.
    """
    if total_num == 0:
        return 0.0
    if not use_ztr:
        return -coef * min(fault_num, 13)
    return -1.0 if fault_num > 0 else 0.0
