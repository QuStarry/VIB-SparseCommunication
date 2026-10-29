"""HumanEval scoring in disposable interpreters, not an adversarial sandbox."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

from .experiment_driver import file_sha256


VERSION = "humaneval_isolated_seeded_v1"
WORKER = r"""
import json
import os
import random
import sys
exit_process = os._exit
try:
    payload = json.load(sys.stdin)
    random.seed(payload["seed"])
    namespace = {"__name__": "__candidate__"}
    exec(payload["code"], namespace)
    exec("\n".join(payload["task"]["test_cases"]), namespace)
    namespace["check"](namespace[payload["task"]["entry_point"]])
except BaseException:
    exit_process(43)
exit_process(42)
"""


def _validate_settings(seed, timeout):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Evaluation seed must be an integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Evaluation timeout must be finite and positive")


def _validate_task(task):
    entry_point = task.get("entry_point")
    tests = task.get("test_cases")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("A single Python entry point is required")
    if not isinstance(tests, list) or not tests or not all(isinstance(test, str) and test for test in tests):
        raise ValueError("Nonempty frozen HumanEval tests are required")


def sanitize_code(output):
    """Match historical code-fence extraction; no hidden-test-driven repair."""
    if not isinstance(output, str):
        raise ValueError("Model output must be text")
    code = output.strip()
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()
    return code


def _kill(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _score_once(code, task, seed, timeout):
    payload = json.dumps({"code": code, "task": task, "seed": seed}).encode("utf-8")
    with TemporaryDirectory(prefix="vib-code-evaluation-") as directory:
        with subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=directory, env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            start_new_session=os.name == "posix",
        ) as process:
            try:
                process.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill(process)
                process.communicate()
                return False
            except BaseException:
                _kill(process)
                process.communicate()
                raise
            if os.name == "posix":
                _kill(process)
            if process.returncode not in (42, 43):
                raise RuntimeError("Evaluation child exited outside the scoring protocol")
            return process.returncode == 42


def score_code(output, task, *, seed=20261028, timeout=5.0):
    """Run two independent, identically seeded checks; disagreement is an error.

    Generated code and tests execute only in children. Assertions, ordinary
    exceptions, SystemExit and timeouts fail; unexpected worker exits halt the
    caller. This is process/state isolation, not filesystem or network sandboxing.
    """
    _validate_settings(seed, timeout)
    _validate_task(task)
    code = sanitize_code(output)
    first = _score_once(code, task, seed, timeout)
    second = _score_once(code, task, seed, timeout)
    if first != second:
        raise RuntimeError("Independent fixed-seed checks are inconsistent")
    return first


def evaluation_protocol(*, seed=20261028, timeout=5.0):
    """Freeze implementation, runtime and scoring settings with the test bundle."""
    _validate_settings(seed, timeout)
    return {
        "version": VERSION, "implementation_sha256": file_sha256(__file__),
        "python_version": sys.version, "seed": seed, "timeout_seconds": timeout,
        "independent_checks": 2, "environment": "minimal_no_inherited_credentials",
        "hash_randomization": "interpreter_default_not_fixed_by_python_random_seed",
        "scope": "process_isolation_not_security_sandbox",
    }


def load_evaluators(bundle_path, expected_sha256):
    """Bind local adapters to the frozen artifact and implementation fingerprint."""
    path = Path(bundle_path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Evaluation bundle fingerprint changed")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    protocol = evaluation_protocol(seed=bundle["seed"], timeout=bundle["timeout_seconds"])
    if any(bundle.get(key) != value for key, value in protocol.items()):
        raise ValueError("Evaluation implementation or protocol differs")
    tasks = bundle["tasks"]
    ids = [task["task_id"] for task in tasks]
    if not tasks or len(set(ids)) != len(ids):
        raise ValueError("Duplicate or empty evaluation task set")
    for task in tasks:
        _validate_task(task)

    def bind(task):
        def evaluate(output):
            if file_sha256(path) != expected_sha256:
                raise ValueError("Evaluation bundle changed before scoring")
            if file_sha256(__file__) != protocol["implementation_sha256"]:
                raise ValueError("Evaluation implementation changed before scoring")
            return score_code(output, task, seed=bundle["seed"], timeout=bundle["timeout_seconds"])
        return evaluate

    return {task["task_id"]: bind(task) for task in tasks}
