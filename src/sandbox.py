"""Execution of model-written code in a separate process with time and memory
limits. Every test case is stored as a pair (argument_list, expected_output) of
real Python objects, so the same worker serves three purposes: scoring visible
cases, scoring hidden cases, and generating expected outputs from the canonical
solution."""
import json
import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

# Script executed in the child process. Mode "check" compares candidate output
# with the expected output for every case; mode "generate" just records outputs.
CASE_WORKER_SCRIPT = r'''
import cmath, copy, json, os, pickle, signal, sys, traceback

class CaseTimeout(Exception):
    pass

def raise_timeout(signal_number, stack_frame):
    raise CaseTimeout("case exceeded the time limit")

def outputs_match(actual_value, expected_value, absolute_tolerance):
    if isinstance(actual_value, bool) or isinstance(expected_value, bool):
        return actual_value == expected_value
    numeric_types = (int, float, complex)
    if isinstance(actual_value, numeric_types) and isinstance(expected_value, numeric_types):
        if isinstance(actual_value, int) and isinstance(expected_value, int):
            return actual_value == expected_value
        both_nan = actual_value != actual_value and expected_value != expected_value
        return both_nan or cmath.isclose(actual_value, expected_value, rel_tol=1e-6,
                                         abs_tol=max(absolute_tolerance, 1e-6))
    if isinstance(actual_value, (list, tuple)) and isinstance(expected_value, (list, tuple)):
        return (type(actual_value) is type(expected_value)
                and len(actual_value) == len(expected_value)
                and all(outputs_match(actual_item, expected_item, absolute_tolerance)
                        for actual_item, expected_item in zip(actual_value, expected_value)))
    if isinstance(actual_value, dict) and isinstance(expected_value, dict):
        return (set(actual_value.keys()) == set(expected_value.keys())
                and all(outputs_match(actual_value[key], expected_value[key], absolute_tolerance)
                        for key in expected_value))
    try:
        return bool(actual_value == expected_value)
    except BaseException:
        return False

def safe_repr(value):
    try:
        return repr(value)[:300]
    except BaseException:
        return "<repr failed: value too large>"

def call_with_timeout(function, argument_list, per_case_timeout):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(per_case_timeout)
    try:
        return function(*copy.deepcopy(argument_list))
    finally:
        signal.alarm(0)

def check_one_case(function, load_error, load_traceback, argument_list, expected_output,
                   test_text, absolute_tolerance, per_case_timeout):
    outcome = {"test": test_text, "passed": False, "expected": safe_repr(expected_output),
               "actual": None, "error_type": None, "error_message": None, "traceback": None}
    if load_error is not None:
        outcome.update(error_type="LoadError", error_message=load_error, traceback=load_traceback)
        return outcome
    if function is None:
        outcome.update(error_type="NameError", error_message="NameError: required function is not defined",
                       traceback="NameError: required function is not defined")
        return outcome
    try:
        actual_output = call_with_timeout(function, argument_list, per_case_timeout)
        outcome["actual"] = safe_repr(actual_output)
        outcome["passed"] = bool(outputs_match(actual_output, expected_output, absolute_tolerance))
        if not outcome["passed"]:
            outcome["error_type"] = "AssertionError"
            outcome["error_message"] = "AssertionError"
            outcome["traceback"] = ("Traceback (most recent call last):\n"
                                    "  File \"<tests>\", line 1, in <module>\n"
                                    "    " + test_text + "\nAssertionError")
    except BaseException as error:
        outcome["error_type"] = type(error).__name__
        outcome["error_message"] = (type(error).__name__ + ": " + str(error))[:300]
        outcome["traceback"] = traceback.format_exc()[-1500:]
    return outcome

def main():
    with open(sys.argv[1], "rb") as payload_file:
        payload = pickle.load(payload_file)
    sys.stdout = open(os.devnull, "w")          # candidate prints must not pollute anything
    namespace, load_error, load_traceback = None, None, None
    try:
        namespace = {"__name__": "__candidate__"}
        exec(compile(payload["solution_code"], "<solution>", "exec"), namespace)
    except BaseException as error:
        load_error = (type(error).__name__ + ": " + str(error))[:500]
        load_traceback = traceback.format_exc()[-1500:]
    function = namespace.get(payload["entry_point"]) if namespace is not None else None

    if payload["mode"] == "generate":
        results = []
        for argument_list in payload["argument_lists"]:
            if function is None:
                results.append((False, None))
                continue
            try:
                output_value = call_with_timeout(function, argument_list, payload["per_case_timeout"])
                pickle.dumps(output_value)       # make sure it can be stored
                results.append((True, output_value))
            except BaseException:
                results.append((False, None))
        with open(payload["result_path"], "wb") as result_file:
            pickle.dump(results, result_file)
    else:
        outcomes = []
        for (argument_list, expected_output), test_text in zip(payload["cases"], payload["test_texts"]):
            try:
                outcomes.append(check_one_case(function, load_error, load_traceback, argument_list,
                                               expected_output, test_text, payload["absolute_tolerance"],
                                               payload["per_case_timeout"]))
            except BaseException as error:   # never let one bad case kill the whole run
                outcomes.append({"test": test_text, "passed": False, "expected": None, "actual": None,
                                 "error_type": type(error).__name__,
                                 "error_message": (type(error).__name__ + ": " + str(error))[:300],
                                 "traceback": None})
        with open(payload["result_path"], "w") as result_file:
            json.dump(outcomes, result_file)

main()
'''


@dataclass
class TestReport:
    """Per-case results of running one candidate on a list of test cases."""
    outcomes: List[dict] = field(default_factory=list)
    fatal_error: Optional[str] = None            # timeout / crash of the whole run

    @property
    def total_count(self) -> int:
        return len(self.outcomes)

    @property
    def passed_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome["passed"])

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_count if self.total_count else 0.0

    @property
    def all_passed(self) -> bool:
        return self.total_count > 0 and self.passed_count == self.total_count


@dataclass
class SubprocessOutcome:
    """Raw outcome of a child Python process."""
    return_code: int
    standard_output: str
    standard_error: str
    timed_out: bool


def _build_resource_limiter(memory_limit_megabytes: int):
    """Create a function that caps the child's address space.

    Input:  memory_limit_megabytes (int).
    Output: callable to pass as `preexec_fn` to subprocess.run.
    """
    def apply_limits():
        import resource
        limit_in_bytes = memory_limit_megabytes * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, limit_in_bytes))
    return apply_limits


def run_python_script(script_path: str, extra_arguments: List[str], time_limit_seconds: int,
                      memory_limit_megabytes: int, working_directory: str) -> SubprocessOutcome:
    """Run a Python script in a child process with time and memory limits.

    Input:  script_path (str), extra_arguments (List[str]), time_limit_seconds (int),
            memory_limit_megabytes (int), working_directory (str).
    Output: SubprocessOutcome (return code, truncated stdout/stderr, timeout flag).
    """
    try:
        completed_process = subprocess.run(
            [sys.executable, script_path, *extra_arguments], capture_output=True, text=True,
            timeout=time_limit_seconds, cwd=working_directory,
            env={**os.environ, "PYTHONHASHSEED": "0"},   # fixed hash seed -> reproducible set/dict order
            preexec_fn=_build_resource_limiter(memory_limit_megabytes))
        return SubprocessOutcome(completed_process.returncode, completed_process.stdout[-2000:],
                                 completed_process.stderr[-2000:], False)
    except subprocess.TimeoutExpired:
        return SubprocessOutcome(-1, "", "TimeoutExpired", True)


def render_test_text(entry_point: str, argument_list: Sequence[Any], expected_output: Any,
                     max_length: int = 600) -> str:
    """Write a test case as an assert statement (for prompts and feedback text).

    Input:  entry_point (str) - function name, argument_list (sequence) - call arguments,
            expected_output (Any), max_length (int) - longer text is truncated.
    Output: str such as "assert add(2, 3) == 5".
    """
    argument_text = ", ".join(repr(argument) for argument in argument_list)
    test_text = f"assert {entry_point}({argument_text}) == {expected_output!r}"
    return test_text if len(test_text) <= max_length else test_text[:max_length] + " ..."


def _run_worker(payload: dict, total_timeout_seconds: int, memory_limit_megabytes: int,
                reads_pickle_result: bool) -> Tuple[Any, SubprocessOutcome]:
    """Write payload + worker to a temp dir, run it, and read back the result file.

    Input:  payload (dict), total_timeout_seconds (int), memory_limit_megabytes (int),
            reads_pickle_result (bool) - True for mode "generate", False for JSON "check".
    Output: (result or None if the worker produced no file, SubprocessOutcome).
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        worker_path = os.path.join(temporary_directory, "worker.py")
        payload_path = os.path.join(temporary_directory, "payload.pkl")
        result_path = os.path.join(temporary_directory, "result.out")
        payload["result_path"] = result_path
        with open(worker_path, "w") as worker_file:
            worker_file.write(CASE_WORKER_SCRIPT)
        with open(payload_path, "wb") as payload_file:
            pickle.dump(payload, payload_file)
        subprocess_outcome = run_python_script(worker_path, [payload_path], total_timeout_seconds,
                                               memory_limit_megabytes, temporary_directory)
        if not os.path.exists(result_path):
            return None, subprocess_outcome
        if reads_pickle_result:
            with open(result_path, "rb") as result_file:
                return pickle.load(result_file), subprocess_outcome
        with open(result_path) as result_file:
            return json.load(result_file), subprocess_outcome


def evaluate_cases(solution_code: str, entry_point: str, cases: List[Tuple[list, Any]],
                   absolute_tolerance: float = 0.0, per_case_timeout_seconds: int = 5,
                   total_timeout_seconds: int = 60, memory_limit_megabytes: int = 4096) -> TestReport:
    """Run candidate code on test cases and compare with the expected outputs.

    Input:  solution_code (str), entry_point (str) - function to call,
            cases (List[(argument_list, expected_output)]), absolute_tolerance (float),
            per_case_timeout_seconds (int), total_timeout_seconds (int),
            memory_limit_megabytes (int).
    Output: TestReport with expected/actual/traceback for every case.
    """
    test_texts = [render_test_text(entry_point, argument_list, expected_output)
                  for argument_list, expected_output in cases]
    payload = {"mode": "check", "solution_code": solution_code, "entry_point": entry_point,
               "cases": cases, "test_texts": test_texts, "absolute_tolerance": absolute_tolerance,
               "per_case_timeout": per_case_timeout_seconds}
    outcomes, subprocess_outcome = _run_worker(payload, total_timeout_seconds,
                                               memory_limit_megabytes, reads_pickle_result=False)
    if outcomes is not None:
        return TestReport(outcomes=outcomes)
    failure_reason = "Timeout" if subprocess_outcome.timed_out else "Crash"
    failed_outcomes = [{"test": test_text, "passed": False, "expected": None, "actual": None,
                        "error_type": failure_reason,
                        "error_message": subprocess_outcome.standard_error[-300:],
                        "traceback": subprocess_outcome.standard_error[-1500:]}
                       for test_text in test_texts]
    return TestReport(outcomes=failed_outcomes, fatal_error=failure_reason)


def compute_expected_outputs(canonical_solution: str, entry_point: str, argument_lists: List[list],
                             per_case_timeout_seconds: int = 10, total_timeout_seconds: int = 300,
                             memory_limit_megabytes: int = 4096) -> List[Tuple[bool, Any]]:
    """Run the canonical solution to obtain the expected output of each input.

    Input:  canonical_solution (str), entry_point (str), argument_lists (List[list]),
            per_case_timeout_seconds (int), total_timeout_seconds (int),
            memory_limit_megabytes (int).
    Output: List[(succeeded, output)] aligned with argument_lists; (False, None)
            for inputs that raised, timed out, or whose output cannot be pickled.
    """
    payload = {"mode": "generate", "solution_code": canonical_solution, "entry_point": entry_point,
               "argument_lists": argument_lists, "per_case_timeout": per_case_timeout_seconds}
    results, _ = _run_worker(payload, total_timeout_seconds, memory_limit_megabytes,
                             reads_pickle_result=True)
    if results is None:
        return [(False, None)] * len(argument_lists)
    return results
