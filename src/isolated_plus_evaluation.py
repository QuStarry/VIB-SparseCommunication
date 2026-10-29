"""Evaluate exported expanded tests without executing candidates in the parent."""

from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .experiment_driver import file_sha256
from .isolated_code_evaluation import _kill, _validate_settings, sanitize_code


WORKER = r"""
import json
import os
import random
import sys
exit_process = os._exit
try:
    payload = json.load(sys.stdin)
    sys.path.insert(0, payload["numpy_path"])
    import numpy as np
    random.seed(payload["seed"])
    np.random.seed(payload["seed"] % (2 ** 32))
    namespace = {"__name__": "__candidate__"}
    exec(payload["code"], namespace)
    exec(payload["task"]["test"], namespace)
    namespace["check"](namespace[payload["task"]["entry_point"]])
except AssertionError:
    exit_process(43)
except BaseException:
    exit_process(44)
exit_process(42)
"""

STATUSES = {42: "pass", 43: "assertion_failure", 44: "runtime_exception"}


def plus_protocol(*, seed=20261029, timeout=30.0):
    """Bind runtime and the explicitly allowlisted NumPy installation."""
    _validate_settings(seed, timeout)
    spec = find_spec("numpy")
    if spec is None or spec.origin is None:
        raise ImportError("Expanded tests require an installed NumPy")
    package = Path(spec.origin).resolve().parent
    return {
        "version": "publisher_exported_plus_isolated_v1", "seed": seed,
        "timeout_seconds": timeout, "independent_checks": 2,
        "implementation_sha256": file_sha256(__file__),
        "isolation_dependency_sha256": file_sha256(Path(__file__).with_name("isolated_code_evaluation.py")),
        "python_version": sys.version, "python_executable": str(Path(sys.executable).resolve()),
        "site_initialization": False,
        "numpy": {"version": version("numpy"), "dependency_path": str(package.parent),
                  "init_sha256": file_sha256(package / "__init__.py")},
        "environment": "minimal_no_inherited_credentials_single_thread_math",
        "timeout_scope": "overall_worker_wall_time_not_evalplus_reference_scaled_per_input",
        "scope": "process_isolation_not_adversarial_filesystem_or_network_sandbox",
        "hash_randomization": "default_interpreter_randomization_not_fixed_by_random_seed",
    }


def _once(code, task, settings):
    payload = json.dumps({
        "code": code, "task": task, "seed": settings["seed"],
        "numpy_path": settings["numpy"]["dependency_path"],
    }).encode("utf-8")
    with TemporaryDirectory(prefix="vib-expanded-evaluation-") as directory:
        with subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", WORKER], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=directory,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            start_new_session=os.name == "posix",
        ) as process:
            try:
                process.communicate(payload, timeout=settings["timeout_seconds"])
            except subprocess.TimeoutExpired:
                _kill(process)
                process.communicate()
                return "timeout"
            except BaseException:
                _kill(process)
                process.communicate()
                raise
            if os.name == "posix":
                _kill(process)
            if process.returncode not in STATUSES:
                raise RuntimeError("Expanded evaluation child exited outside the scoring protocol")
            return STATUSES[process.returncode]


def score_plus(output, task, *, seed=20261029, timeout=30.0):
    """Run the unmodified exported check in two disposable fixed-seed workers."""
    if (not isinstance(task.get("entry_point"), str) or not task["entry_point"].isidentifier()
            or not isinstance(task.get("test"), str) or not task["test"].strip()):
        raise ValueError("A valid entry point and nonempty exported test are required")
    settings = plus_protocol(seed=seed, timeout=timeout)
    code = sanitize_code(output)
    first = _once(code, task, settings)
    second = _once(code, task, settings)
    if first != second:
        raise RuntimeError("Independent expanded checks are inconsistent")
    return {"passed": first == "pass", "status": first, "independent_checks": 2}
