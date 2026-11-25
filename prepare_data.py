"""Build the datasets used by the experiments from EvalPlus's get_mbpp_plus().

    python prepare_data.py --data_directory data --eval_fraction 0.10 --seed 0

Writes:
  data/raw/mbpp_plus_raw.pkl, data/raw/raw_manifest.json   exact EvalPlus output
  data/processed/dev_problems.jsonl, dev_visible_cases.pkl  dev split (model-facing)
  data/private/{dev,eval}_hidden_cases.pkl                  unseen test cases
  data/private/eval_problems.jsonl, eval_visible_cases.pkl  held-out eval split
  data/private/canonical_solutions.pkl, split_manifest.json
Visible cases = first N original MBPP inputs. Hidden cases = the remaining
original inputs + all MBPP+ extra inputs (minus any duplicate of a visible case).
Expected outputs are produced by running the canonical solution."""
import argparse
import json
import math
import os
import pickle
import random
from collections import Counter
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

from src.sandbox import compute_expected_outputs, render_test_text


def as_list(possibly_missing_inputs: Any) -> list:
    """Return the value if it is a list, else an empty list.

    Input:  possibly_missing_inputs (Any) - e.g. task Mbpp/793 stores `{}` instead of a list.
    Output: list.
    """
    # Some EvalPlus records contain placeholders instead of a list.
    return possibly_missing_inputs if isinstance(possibly_missing_inputs, list) else []


def clean_task_description(raw_prompt: str) -> str:
    """Strip quotes and example asserts from the raw EvalPlus prompt.

    Input:  raw_prompt (str) - the 'prompt' field of an MBPP+ problem.
    Output: str - plain task description without test lines.
    """
    # Remove triple quotes and assert statements so the model receives only the task text.
    description_lines = [line for line in raw_prompt.replace('"""', "").strip().splitlines()
                         if not line.strip().startswith("assert")]
    return "\n".join(description_lines).strip()


def value_round_trips(value: Any) -> bool:
    """Check that repr(value) can be evaluated back to an equal value.

    Input:  value (Any).
    Output: bool - True if the assert text shown to the model faithfully shows the value.
    """
    try:
        # Re-evaluate repr(value) to ensure rendered test values can be read back safely.
        restored_value = eval(repr(value), {"inf": float("inf"), "nan": float("nan")})
    except Exception:
        return False
    if isinstance(value, float) and math.isnan(value):
        return isinstance(restored_value, float) and math.isnan(restored_value)
    try:
        return bool(restored_value == value) and type(restored_value) is type(value)
    except Exception:
        return False


def prepare_task(record: dict, num_visible_tests: int, per_case_timeout_seconds: int,
                 max_case_bytes: int = 1_000_000) -> Tuple[Optional[dict], str]:
    """Turn one EvalPlus problem into model-facing text, visible cases and hidden cases.

    Input:  record (dict) - one value of get_mbpp_plus(), num_visible_tests (int),
            per_case_timeout_seconds (int) for running the canonical solution,
            max_case_bytes (int) - hidden cases whose pickled inputs+output exceed this
            are dropped (some MBPP+ outputs are hundreds of MB and crash the sandbox).
    Output: (prepared dict or None, reason) - reason is 'ok' or why the task was skipped.
    """
    base_inputs = as_list(record["base_input"])
    plus_inputs = as_list(record["plus_input"])
    # The original base inputs supply the visible tests, so enough must exist before splitting.
    if len(base_inputs) < num_visible_tests:
        return None, "too_few_base_inputs"

    all_inputs = base_inputs + plus_inputs
    # Execute each canonical solution once so visible and hidden outputs come from the same source.
    canonical_results = compute_expected_outputs(record["canonical_solution"], record["entry_point"],
                                                 all_inputs, per_case_timeout_seconds)
    visible_results = canonical_results[:num_visible_tests]
    if not all(succeeded for succeeded, _ in visible_results):
        print(visible_results)
        return None, "canonical_failed_on_visible_input"

    visible_cases = [(all_inputs[position], visible_results[position][1]) for position in range(num_visible_tests)]
    visible_tests = [render_test_text(record["entry_point"], argument_list, expected_output, max_length=10**6)
                     for argument_list, expected_output in visible_cases]
    if any(len(test_text) > 600 for test_text in visible_tests):
        return None, "visible_test_too_long"
    if not all(value_round_trips(argument) for argument_list, _ in visible_cases for argument in argument_list) \
            or not all(value_round_trips(expected_output) for _, expected_output in visible_cases):
        return None, "visible_test_not_printable"

    visible_signatures = {repr(argument_list) for argument_list, _ in visible_cases}
    hidden_cases = []
    num_oversized_cases = 0
    # Only inputs after the visible prefix are candidates for hidden testing.
    for position in range(num_visible_tests, len(all_inputs)):
        succeeded, expected_output = canonical_results[position]
        if not succeeded or repr(all_inputs[position]) in visible_signatures:
            continue
        # Skip unusually large cases that could exceed sandbox or serialization limits.
        case_size_in_bytes = len(pickle.dumps((all_inputs[position], expected_output)))
        if case_size_in_bytes > max_case_bytes:
            num_oversized_cases += 1
            continue
        hidden_cases.append((all_inputs[position], expected_output))
    if not hidden_cases:
        return None, "no_hidden_cases"

    problem_record = {"task_id": record["task_id"],
                      "description": clean_task_description(record["prompt"]),
                      "function_name": record["entry_point"],
                      "absolute_tolerance": float(record.get("atol", 0) or 0),
                      "visible_tests": visible_tests, "num_hidden_cases": len(hidden_cases),
                      "num_oversized_cases_dropped": num_oversized_cases}
    return {"problem": problem_record, "visible_cases": visible_cases, "hidden_cases": hidden_cases}, "ok"


def write_pickle(path: str, python_object: Any) -> None:
    """Save an object with pickle (exact types). Input: path (str), python_object. Output: None."""
    # Create the split-specific parent directory before writing.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as output_file:
        pickle.dump(python_object, output_file)


def write_jsonl(path: str, records: List[dict]) -> None:
    """Save plain-text records, one JSON object per line. Input: path (str), records. Output: None."""
    # JSONL lets consumers load one problem record at a time.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def main() -> None:
    """Download via EvalPlus, prepare every task, split 90/10 and write all folders."""
    from evalplus.data import get_mbpp_plus

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", default="data")
    parser.add_argument("--num_visible_tests", type=int, default=3)
    parser.add_argument("--eval_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per_case_timeout_seconds", type=int, default=10)
    parser.add_argument("--max_case_bytes", type=int, default=1_000_000)
    arguments = parser.parse_args()
    data_directory = arguments.data_directory

    # Keep an exact copy of EvalPlus output before filtering or splitting tasks.
    raw_problems: Dict[str, dict] = get_mbpp_plus()
    write_pickle(os.path.join(data_directory, "raw", "mbpp_plus_raw.pkl"), raw_problems)
    with open(os.path.join(data_directory, "raw", "raw_manifest.json"), "w") as manifest_file:
        json.dump({"source": "evalplus.data.get_mbpp_plus()", "evalplus_version": metadata.version("evalplus"),
                   "num_problems": len(raw_problems)}, manifest_file, indent=2)

    prepared_by_task: Dict[str, dict] = {}
    skip_reasons: Counter = Counter()
    skipped_tasks: Dict[str, str] = {}
    for position, (task_id, record) in enumerate(sorted(raw_problems.items())):
        prepared_task, reason = prepare_task(record, arguments.num_visible_tests,
                                              arguments.per_case_timeout_seconds, arguments.max_case_bytes)
        if prepared_task is None:
            skip_reasons[reason] += 1
            skipped_tasks[task_id] = reason
        else:
            prepared_by_task[task_id] = prepared_task
        if (position + 1) % 50 == 0:
            print(f"prepared {position + 1}/{len(raw_problems)}", flush=True)

    kept_task_ids = sorted(prepared_by_task)
    # Shuffle in a reproducible order, then place the leading tasks in the eval split.
    random.Random(arguments.seed).shuffle(kept_task_ids)
    num_eval_tasks = max(1, round(arguments.eval_fraction * len(kept_task_ids)))
    split_task_ids = {"eval": sorted(kept_task_ids[:num_eval_tasks]), "dev": sorted(kept_task_ids[num_eval_tasks:])}

    for split_name, task_ids in split_task_ids.items():
        model_facing_folder = "processed" if split_name == "dev" else "private"
        write_jsonl(os.path.join(data_directory, model_facing_folder, f"{split_name}_problems.jsonl"),
                    [prepared_by_task[task_id]["problem"] for task_id in task_ids])
        write_pickle(os.path.join(data_directory, model_facing_folder, f"{split_name}_visible_cases.pkl"),
                     {task_id: prepared_by_task[task_id]["visible_cases"] for task_id in task_ids})
        write_pickle(os.path.join(data_directory, "private", f"{split_name}_hidden_cases.pkl"),
                     {task_id: prepared_by_task[task_id]["hidden_cases"] for task_id in task_ids})
    write_pickle(os.path.join(data_directory, "private", "canonical_solutions.pkl"),
                 {task_id: raw_problems[task_id]["canonical_solution"] for task_id in kept_task_ids})
    with open(os.path.join(data_directory, "private", "split_manifest.json"), "w") as manifest_file:
        json.dump({"seed": arguments.seed, "eval_fraction": arguments.eval_fraction,
                   "num_visible_tests": arguments.num_visible_tests, "dev_task_ids": split_task_ids["dev"],
                   "eval_task_ids": split_task_ids["eval"], "skipped_tasks": skipped_tasks}, manifest_file, indent=2)

    print(f"raw problems: {len(raw_problems)} | kept: {len(kept_task_ids)} | "
          f"dev: {len(split_task_ids['dev'])} | eval: {len(split_task_ids['eval'])}")
    print(f"skipped by reason: {dict(skip_reasons)}")
    print("oversized hidden cases dropped: "
          f"{sum(item['problem']['num_oversized_cases_dropped'] for item in prepared_by_task.values())}")


if __name__ == "__main__":
    main()
