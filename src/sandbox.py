"""Safe-ish execution of model-written code in a separate process with time and
memory limits. Provides (a) per-test results on VISIBLE tests with expected vs.
actual values, and (b) a pass/fail verdict on the HIDDEN test script."""
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

# Script executed inside the child process. It loads the candidate code, runs
# each visible assert individually and records expected/actual/traceback.
# Results go to a JSON file (not stdout) because candidate code may print.
VISIBLE_TEST_HARNESS = r'''
import ast, json, math, os, signal, sys, traceback

class TestTimeout(Exception):
    pass

def raise_timeout(signal_number, stack_frame):
    raise TestTimeout("test exceeded the time limit")

def values_match(actual_value, expected_value):
    if isinstance(actual_value, float) or isinstance(expected_value, float):
        try:
            return math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-6)
        except TypeError:
            pass
    return actual_value == expected_value

def run_single_test(test_text, namespace, per_test_timeout):
    outcome = {"test": test_text, "passed": False, "expected": None, "actual": None,
               "error_type": None, "error_message": None, "traceback": None}
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(per_test_timeout)
    try:
        statement = ast.parse(test_text.strip()).body[0]
        is_simple_equality = (
            isinstance(statement, ast.Assert)
            and isinstance(statement.test, ast.Compare)
            and len(statement.test.ops) == 1
            and isinstance(statement.test.ops[0], ast.Eq))
        if is_simple_equality:
            left_expression = ast.fix_missing_locations(ast.Expression(statement.test.left))
            right_expression = ast.fix_missing_locations(ast.Expression(statement.test.comparators[0]))
            actual_value = eval(compile(left_expression, "<tests>", "eval"), namespace)
            expected_value = eval(compile(right_expression, "<tests>", "eval"), namespace)
            outcome["actual"] = repr(actual_value)[:300]
            outcome["expected"] = repr(expected_value)[:300]
            outcome["passed"] = bool(values_match(actual_value, expected_value))
            if not outcome["passed"]:
                outcome["error_type"] = "AssertionError"
                outcome["error_message"] = "AssertionError"
                outcome["traceback"] = ("Traceback (most recent call last):\n"
                                        "  File \"<tests>\", line 1, in <module>\n"
                                        "    " + test_text.strip() + "\nAssertionError")
        else:
            exec(compile(ast.parse(test_text.strip()), "<tests>", "exec"), namespace)
            outcome["passed"] = True
    except BaseException as error:
        outcome["passed"] = False
        outcome["error_type"] = type(error).__name__
        outcome["error_message"] = (type(error).__name__ + ": " + str(error))[:300]
        outcome["traceback"] = traceback.format_exc()[-1500:]
    finally:
        signal.alarm(0)
    return outcome

def main():
    payload = json.load(open(sys.argv[1]))
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    namespace = {"__name__": "__candidate__"}
    load_error = None
    try:
        exec(compile(payload["solution_code"], "<solution>", "exec"), namespace)
    except BaseException as error:
        load_error = (type(error).__name__ + ": " + str(error))[:500]
        load_traceback = traceback.format_exc()[-1500:]
    outcomes = []
    for test_text in payload["tests"]:
        if load_error is not None:
            outcomes.append({"test": test_text, "passed": False, "expected": None, "actual": None,
                             "error_type": "LoadError", "error_message": load_error,
                             "traceback": load_traceback})
        else:
            outcomes.append(run_single_test(test_text, namespace, payload["per_test_timeout"]))
    sys.stdout = real_stdout
    json.dump({"outcomes": outcomes}, open(payload["result_path"], "w"))

main()
'''


@dataclass
class VisibleTestReport:
    """Result of running the visible asserts against one candidate."""
    outcomes: List[dict] = field(default_factory=list)  # one dict per visible test
    fatal_error: Optional[str] = None                   # timeout / crash of the whole run

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
    Output: callable to be passed as `preexec_fn` to subprocess.run.
    """
    def apply_limits():
        import resource
        limit_in_bytes = memory_limit_megabytes * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes, limit_in_bytes))
    return apply_limits


def run_python_script(script_path: str, extra_arguments: List[str], time_limit_seconds: int,
                      memory_limit_megabytes: int, working_directory: str) -> SubprocessOutcome:
    """Run a Python script in a child process with time and memory limits.

    Input:  script_path (str), extra_arguments (List[str]) - argv for the script,
            time_limit_seconds (int), memory_limit_megabytes (int),
            working_directory (str) - cwd of the child.
    Output: SubprocessOutcome (return code, truncated stdout/stderr, timeout flag).
    """
    try:
        completed_process = subprocess.run(
            [sys.executable, script_path, *extra_arguments],
            capture_output=True, text=True, timeout=time_limit_seconds, cwd=working_directory,
            preexec_fn=_build_resource_limiter(memory_limit_megabytes))
        return SubprocessOutcome(completed_process.returncode, completed_process.stdout[-2000:],
                                 completed_process.stderr[-2000:], False)
    except subprocess.TimeoutExpired:
        return SubprocessOutcome(-1, "", "TimeoutExpired", True)


def evaluate_visible_tests(solution_code: str, visible_tests: List[str],
                           per_test_timeout_seconds: int = 5,
                           memory_limit_megabytes: int = 4096) -> VisibleTestReport:
    """Run candidate code against the visible asserts, one by one.

    Input:  solution_code (str)    - candidate Python source,
            visible_tests (List[str]) - assert statements,
            per_test_timeout_seconds (int), memory_limit_megabytes (int).
    Output: VisibleTestReport with per-test expected/actual/traceback details.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        harness_path = os.path.join(temporary_directory, "harness.py")
        payload_path = os.path.join(temporary_directory, "payload.json")
        result_path = os.path.join(temporary_directory, "result.json")
        with open(harness_path, "w") as harness_file:
            harness_file.write(VISIBLE_TEST_HARNESS)
        with open(payload_path, "w") as payload_file:
            json.dump({"solution_code": solution_code, "tests": visible_tests,
                       "per_test_timeout": per_test_timeout_seconds,
                       "result_path": result_path}, payload_file)

        total_time_limit = per_test_timeout_seconds * len(visible_tests) + 10
        subprocess_outcome = run_python_script(harness_path, [payload_path], total_time_limit,
                                               memory_limit_megabytes, temporary_directory)
        if os.path.exists(result_path):
            with open(result_path) as result_file:
                return VisibleTestReport(outcomes=json.load(result_file)["outcomes"])

        failure_reason = "Timeout" if subprocess_outcome.timed_out else "Crash"
        failed_outcomes = [
            {"test": test_text, "passed": False, "expected": None, "actual": None,
             "error_type": failure_reason, "error_message": subprocess_outcome.standard_error[-300:],
             "traceback": subprocess_outcome.standard_error[-1500:]}
            for test_text in visible_tests]
        return VisibleTestReport(outcomes=failed_outcomes, fatal_error=failure_reason)


def evaluate_hidden_tests(solution_code: str, hidden_test_script: str,
                          time_limit_seconds: int = 60, memory_limit_megabytes: int = 4096) -> bool:
    """Run candidate code followed by the hidden test script.

    Input:  solution_code (str), hidden_test_script (str) - standalone script that
            raises on failure, time_limit_seconds (int), memory_limit_megabytes (int).
    Output: bool - True only if the process exits with code 0 within the time limit.
    """
    with tempfile.TemporaryDirectory() as temporary_directory:
        script_path = os.path.join(temporary_directory, "hidden_check.py")
        with open(script_path, "w") as script_file:
            script_file.write(solution_code + "\n\n" + hidden_test_script + "\n")
        subprocess_outcome = run_python_script(script_path, [], time_limit_seconds,
                                               memory_limit_megabytes, temporary_directory)
        return subprocess_outcome.return_code == 0 and not subprocess_outcome.timed_out
