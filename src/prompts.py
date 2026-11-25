"""Prompt construction and answer parsing for three prompting styles:
  base  - answer with code only,
  cot   - reason step by step, then give code,
  react - ReAct format: Thought -> Action (code); feedback arrives as an Observation."""
import re
from typing import Dict, List

from src.data_loading import CodingProblem

PROMPT_STYLES = ["base", "cot", "react"]

SYSTEM_PROMPTS = {
    "base": ("You are an expert Python programmer. Reply with ONLY the complete solution inside one "
             "```python block and no explanation. Write a general solution that is correct for ALL valid "
             "inputs, not only the shown examples."),
    "cot": ("You are an expert Python programmer. Think step by step about the task and its edge cases, "
            "then give the final complete solution inside one ```python block at the end of your reply. "
            "Write a general solution that is correct for ALL valid inputs, not only the shown examples."),
    "react": ("You are an expert Python programmer working in a Thought/Action loop. Reply in this format:\n"
              "Thought: <short reasoning about the task or about the latest observation>\n"
              "Action:\n```python\n<complete, self-contained solution>\n```\n"
              "Write a general solution that is correct for ALL valid inputs, not only the shown examples."),
}

CLOSING_INSTRUCTIONS = {
    "base": "Output the corrected full solution.",
    "cot": "Think step by step about what is wrong, then output the corrected full solution.",
    "react": "Write your next Thought about what is wrong, then output the corrected full solution as the Action.",
}


def format_problem_statement(problem: CodingProblem) -> str:
    """Render the task description, function name and the visible tests.

    Input:  problem (CodingProblem).
    Output: str - text block placed at the top of every prompt.
    """
    visible_tests_text = "\n".join(problem.visible_tests)
    return (f"Task: {problem.description}\n"
            f"Implement the function `{problem.function_name}`.\n"
            f"It must satisfy these example tests:\n{visible_tests_text}")


def build_initial_prompt(problem: CodingProblem, prompt_style: str) -> List[Dict[str, str]]:
    """Chat messages for first-attempt (single-shot / best-of-N) generation.

    Input:  problem (CodingProblem), prompt_style (str) - 'base' | 'cot' | 'react'.
    Output: List[dict] chat messages.
    """
    return [{"role": "system", "content": SYSTEM_PROMPTS[prompt_style]},
            {"role": "user", "content": format_problem_statement(problem)}]


def build_reflection_prompt(problem: CodingProblem, previous_code: str,
                            prompt_style: str) -> List[Dict[str, str]]:
    """Chat messages asking the model to self-review WITHOUT any execution feedback.

    Input:  problem (CodingProblem), previous_code (str), prompt_style (str).
    Output: List[dict] chat messages.
    """
    user_message = (f"{format_problem_statement(problem)}\n\nHere is your previous solution:\n"
                    f"```python\n{previous_code}\n```\n"
                    "Carefully re-read the task and review the solution for bugs or missed edge cases. "
                    "Then output the improved (or unchanged, if already correct) full solution.")
    return [{"role": "system", "content": SYSTEM_PROMPTS[prompt_style]},
            {"role": "user", "content": user_message}]


def build_repair_prompt(problem: CodingProblem, previous_code: str, feedback_text: str,
                        prompt_style: str) -> List[Dict[str, str]]:
    """Chat messages asking the model to fix code given execution feedback.

    Input:  problem (CodingProblem), previous_code (str), feedback_text (str) -
            output of `feedback.build_feedback_text`, prompt_style (str).
    Output: List[dict] chat messages.
    """
    feedback_label = "Observation" if prompt_style == "react" else "Execution feedback"
    user_message = (f"{format_problem_statement(problem)}\n\nYour previous solution:\n"
                    f"```python\n{previous_code}\n```\n\n{feedback_label}:\n{feedback_text}\n\n"
                    f"{CLOSING_INSTRUCTIONS[prompt_style]}")
    return [{"role": "system", "content": SYSTEM_PROMPTS[prompt_style]},
            {"role": "user", "content": user_message}]


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
