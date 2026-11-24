"""Prompt construction and answer parsing. The model is asked to follow a
ReAct-style format: a short 'Thought' followed by the code as the 'Action'."""
import re
from typing import Dict, List

from src.data_loading import CodingProblem

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Reply in this format:\n"
    "Thought: <one short paragraph of reasoning>\n"
    "Code:\n```python\n<complete, self-contained solution>\n```\n"
    "Write a general solution that is correct for ALL valid inputs, not only the shown examples.")


def format_problem_statement(problem: CodingProblem) -> str:
    """Render the task description, function name and the visible tests.

    Input:  problem (CodingProblem).
    Output: str - text block placed at the top of every prompt.
    """
    visible_tests_text = "\n".join(problem.visible_tests)
    return (f"Task: {problem.description}\n"
            f"Implement the function `{problem.function_name}`.\n"
            f"It must satisfy these example tests:\n{visible_tests_text}")


def build_initial_prompt(problem: CodingProblem) -> List[Dict[str, str]]:
    """Chat messages for first-attempt (single-shot / best-of-N) generation.

    Input:  problem (CodingProblem).
    Output: List[dict] chat messages.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_problem_statement(problem)}]


def build_reflection_prompt(problem: CodingProblem, previous_code: str) -> List[Dict[str, str]]:
    """Chat messages asking the model to self-review WITHOUT any execution feedback.

    Input:  problem (CodingProblem), previous_code (str) - the code to review.
    Output: List[dict] chat messages.
    """
    user_message = (f"{format_problem_statement(problem)}\n\nHere is your previous solution:\n"
                    f"```python\n{previous_code}\n```\n"
                    "Carefully re-read the task and review the solution for bugs or missed edge cases. "
                    "Then output the improved (or unchanged, if already correct) full solution.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def build_repair_prompt(problem: CodingProblem, previous_code: str,
                        feedback_text: str) -> List[Dict[str, str]]:
    """Chat messages asking the model to fix code given execution feedback.

    Input:  problem (CodingProblem), previous_code (str), feedback_text (str) -
            output of `feedback.build_feedback_text`.
    Output: List[dict] chat messages.
    """
    user_message = (f"{format_problem_statement(problem)}\n\nYour previous solution:\n"
                    f"```python\n{previous_code}\n```\n\nExecution feedback:\n{feedback_text}\n\n"
                    "Reason about what is wrong (Thought), then output the corrected full solution.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def extract_code_block(model_reply: str) -> str:
    """Pull the last fenced Python block out of a model reply.

    Input:  model_reply (str) - raw model text.
    Output: str - code inside the last ```python block (or any ``` block);
            the whole reply if no fenced block exists.
    """
    python_blocks = re.findall(r"```python\s*\n(.*?)```", model_reply, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()
    generic_blocks = re.findall(r"```\s*\n(.*?)```", model_reply, re.DOTALL)
    if generic_blocks:
        return generic_blocks[-1].strip()
    return model_reply.strip()
