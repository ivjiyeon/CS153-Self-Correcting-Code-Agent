"""Load MBPP+ and split every problem's tests into VISIBLE tests (shown to the
model and used for execution feedback) and HIDDEN tests (used only for final
evaluation, never shown to the model)."""
import ast
import random
import re
from dataclasses import dataclass
from typing import List


@dataclass
class CodingProblem:
    """One MBPP+ problem with its visible / hidden test split."""
    task_id: int
    description: str              # natural-language task statement
    function_name: str            # name the model must implement
    visible_tests: List[str]      # assert statements shown to the model
    hidden_test_script: str       # standalone script; exit code 0 == all hidden tests pass
    reference_solution: str       # ground-truth code (used only for sanity checks)


def clean_task_description(raw_prompt: str) -> str:
    """Strip quotes and example asserts from the raw MBPP+ prompt.

    Input:  raw_prompt (str) - the 'prompt' field of an MBPP+ record.
    Output: str - the plain task description without test lines.
    """
    description_lines = []
    for line in raw_prompt.replace('"""', "").strip().splitlines():
        if line.strip().startswith("assert"):
            continue
        description_lines.append(line)
    return "\n".join(description_lines).strip()


def extract_function_name(reference_solution: str, example_assertion: str) -> str:
    """Find the name of the function under test.

    Input:  reference_solution (str) - ground-truth code,
            example_assertion (str)  - one assert statement calling the function.
    Output: str - the function name that appears both as a `def` and in the assert
            (falls back to the first called name in the assert).
    """
    defined_function_names = re.findall(r"^def\s+(\w+)\s*\(", reference_solution, re.MULTILINE)
    called_function_names = [
        node.func.id
        for node in ast.walk(ast.parse(example_assertion.strip()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    for called_name in called_function_names:
        if called_name in defined_function_names:
            return called_name
    return called_function_names[0] if called_function_names else defined_function_names[-1]


def build_problem(record: dict, num_visible_tests: int) -> CodingProblem:
    """Convert one raw dataset record to a CodingProblem.

    Input:  record (dict) - one row of `evalplus/mbppplus`,
            num_visible_tests (int) - how many asserts to reveal to the model.
    Output: CodingProblem.
    """
    all_base_tests = list(record["test_list"])
    visible_tests = all_base_tests[:num_visible_tests]
    leftover_base_tests = all_base_tests[num_visible_tests:]

    # Hidden evaluation = leftover base asserts + the extended MBPP+ test script.
    imports_block = "\n".join(record.get("test_imports", []) or [])
    hidden_test_script = "\n".join(
        [imports_block, record.get("test", "") or "", "\n".join(leftover_base_tests)]
    )
    return CodingProblem(
        task_id=int(record["task_id"]),
        description=clean_task_description(record["prompt"]),
        function_name=extract_function_name(record["code"], visible_tests[0]),
        visible_tests=visible_tests,
        hidden_test_script=hidden_test_script,
        reference_solution=record["code"],
    )


def load_problems(num_problems: int, num_visible_tests: int, seed: int,
                  shard_index: int = 0, num_shards: int = 1) -> List[CodingProblem]:
    """Sample problems reproducibly and return this shard's slice.

    Input:  num_problems (int)      - total problems to sample (e.g. 200),
            num_visible_tests (int) - visible tests per problem (e.g. 3),
            seed (int)              - sampling seed,
            shard_index / num_shards (int) - split work across SLURM array tasks.
    Output: List[CodingProblem] for this shard.
    """
    from datasets import load_dataset

    dataset = load_dataset("evalplus/mbppplus", split="test")
    all_records = [dataset[row_index] for row_index in range(len(dataset))]
    random.Random(seed).shuffle(all_records)
    selected_records = all_records[:num_problems]
    shard_records = selected_records[shard_index::num_shards]
    return [build_problem(record, num_visible_tests) for record in shard_records]


if __name__ == "__main__":
    # Sanity check: the reference solution must pass both visible and hidden tests.
    from src.sandbox import evaluate_hidden_tests, evaluate_visible_tests

    for sample_problem in load_problems(num_problems=5, num_visible_tests=3, seed=0):
        visible_report = evaluate_visible_tests(sample_problem.reference_solution,
                                                sample_problem.visible_tests)
        hidden_passed = evaluate_hidden_tests(sample_problem.reference_solution,
                                              sample_problem.hidden_test_script)
        print(f"task {sample_problem.task_id} ({sample_problem.function_name}): "
              f"visible {visible_report.passed_count}/{visible_report.total_count}, "
              f"hidden_passed={hidden_passed}")
