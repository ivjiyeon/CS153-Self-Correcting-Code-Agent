"""Load the prepared MBPP+ splits from disk (created by prepare_data.py).

Folder layout:
  data/raw/        exact get_mbpp_plus() output (pickle) + manifest
  data/processed/  dev split (90%): model-facing problem text + visible cases
  data/private/    hidden cases for every problem, the held-out eval split (10%),
                   canonical solutions and the split manifest
Test cases are stored as pickles of real Python objects (tuples, dicts, complex
numbers survive exactly), so no JSON deserialization is needed."""
import argparse
import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

DEFAULT_DATA_DIRECTORY = "data"
SPLIT_NAMES = ["dev", "eval"]


@dataclass
class CodingProblem:
    """One MBPP+ problem with its visible / hidden test split."""
    task_id: str                                   # e.g. "Mbpp/100"
    description: str                               # natural-language task statement
    function_name: str                             # entry point the model must implement
    absolute_tolerance: float                      # float tolerance for output comparison
    visible_tests: List[str]                       # assert texts shown to the model
    visible_cases: List[Tuple[list, Any]]          # (arguments, expected) behind visible_tests
    hidden_cases: List[Tuple[list, Any]]           # unseen cases (rest of base + plus inputs)
    reference_solution: str                        # canonical code, sanity checks only

    @property
    def task_number(self) -> int:
        """Numeric part of the task id (used to seed sampling). Output: int."""
        return int(self.task_id.split("/")[-1])


def get_split_paths(data_directory: str, split_name: str) -> Dict[str, str]:
    """Locate the files of one split.

    Input:  data_directory (str), split_name (str) - 'dev' or 'eval'.
    Output: dict with keys problems, visible_cases, hidden_cases (file paths).
            The dev split's model-facing files are in processed/, the eval split
            lives entirely in private/. Hidden cases are always in private/.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}")
    model_facing_folder = "processed" if split_name == "dev" else "private"
    return {
        "problems": os.path.join(data_directory, model_facing_folder, f"{split_name}_problems.jsonl"),
        "visible_cases": os.path.join(data_directory, model_facing_folder, f"{split_name}_visible_cases.pkl"),
        "hidden_cases": os.path.join(data_directory, "private", f"{split_name}_hidden_cases.pkl"),
    }


def load_problems(split_name: str, num_problems: int, seed: int, shard_index: int = 0,
                  num_shards: int = 1, data_directory: str = DEFAULT_DATA_DIRECTORY) -> List[CodingProblem]:
    """Load a split, sample reproducibly, and return this shard's slice.

    Input:  split_name (str) - 'dev' or 'eval', num_problems (int) - max problems to
            use (<= split size), seed (int), shard_index / num_shards (int) - SLURM
            array sharding, data_directory (str).
    Output: List[CodingProblem] for this shard.
    """
    paths = get_split_paths(data_directory, split_name)
    with open(paths["problems"]) as problems_file:
        problem_records = [json.loads(line) for line in problems_file]
    with open(paths["visible_cases"], "rb") as visible_file:
        visible_cases_by_task = pickle.load(visible_file)
    with open(paths["hidden_cases"], "rb") as hidden_file:
        hidden_cases_by_task = pickle.load(hidden_file)
    with open(os.path.join(data_directory, "private", "canonical_solutions.pkl"), "rb") as solutions_file:
        canonical_solutions = pickle.load(solutions_file)

    random.Random(seed).shuffle(problem_records)
    selected_records = problem_records[:num_problems][shard_index::num_shards]
    return [CodingProblem(
        task_id=record["task_id"], description=record["description"],
        function_name=record["function_name"], absolute_tolerance=record["absolute_tolerance"],
        visible_tests=record["visible_tests"],
        visible_cases=visible_cases_by_task[record["task_id"]],
        hidden_cases=hidden_cases_by_task[record["task_id"]],
        reference_solution=canonical_solutions[record["task_id"]]) for record in selected_records]


if __name__ == "__main__":
    # Sanity check: canonical solutions must pass everything, a stub must fail hidden cases.
    from src.sandbox import evaluate_cases

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--split", default="dev", choices=SPLIT_NAMES)
    argument_parser.add_argument("--limit", type=int, default=10_000)
    argument_parser.add_argument("--data_directory", default=DEFAULT_DATA_DIRECTORY)
    parsed_arguments = argument_parser.parse_args()

    problems = load_problems(parsed_arguments.split, parsed_arguments.limit, seed=0,
                             data_directory=parsed_arguments.data_directory)
    failures = 0
    for problem in problems:
        visible_report = evaluate_cases(problem.reference_solution, problem.function_name,
                                        problem.visible_cases, problem.absolute_tolerance)
        hidden_report = evaluate_cases(problem.reference_solution, problem.function_name,
                                       problem.hidden_cases, problem.absolute_tolerance)
        stub_code = f"def {problem.function_name}(*arguments, **keyword_arguments):\n    return None\n"
        stub_report = evaluate_cases(stub_code, problem.function_name, problem.hidden_cases,
                                     problem.absolute_tolerance)
        reference_ok = visible_report.all_passed and hidden_report.all_passed
        stub_correctly_fails = not stub_report.all_passed
        if not (reference_ok and stub_correctly_fails):
            failures += 1
            print(f"PROBLEM {problem.task_id}: reference_ok={reference_ok} "
                  f"stub_fails={stub_correctly_fails}")
    print(f"Checked {len(problems)} problems in split '{parsed_arguments.split}': {failures} with issues.")
