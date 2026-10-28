"""Offline evidence preflight and isolated, approved single-attempt scheduling.

Evidence review and adapter correctness remain the caller's responsibility.
This module checks their frozen identities, not tokenizer or invoice truth.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .experiment_session import ExperimentSession
from .request_ledger import fingerprint


def file_sha256(path):
    """Hash an evidence file without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def driver_sha256():
    """Identity to freeze before approval; source drift requires a new manifest."""
    return file_sha256(__file__)


def implementation_sha256():
    """Freeze local request, accounting and session implementations as well."""
    root = Path(__file__).parent
    return {
        name: file_sha256(root / name)
        for name in ("experiment_session.py", "inference_protocol.py", "request_ledger.py", "llm_client.py")
    }


def _artifact_valid(evidence):
    try:
        return file_sha256(evidence["path"]) == evidence["sha256"]
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _evidence_blockers(manifest, selected_job=None):
    evidence = manifest.get("preflight", {})
    if not isinstance(evidence, dict):
        return ["missing_preflight_evidence"]
    blockers = []
    if evidence.get("driver_sha256") != driver_sha256():
        blockers.append("driver_source_mismatch")
    if evidence.get("implementation_sha256") != implementation_sha256():
        blockers.append("request_implementation_mismatch")
    endpoint = {name: manifest.get(name) for name in ("provider", "model", "base_url")}
    price = evidence.get("price_evidence")
    if not isinstance(price, dict) or not (
        _artifact_valid(price)
        and price.get("endpoint") == endpoint
        and price.get("price") == manifest.get("price")
        and price.get("reviewed_upper_bound") is True
    ):
        blockers.append("unverified_price_evidence")
    evaluation = evidence.get("evaluation_evidence")
    if not isinstance(evaluation, dict) or not (
        _artifact_valid(evaluation) and evaluation.get("version") == manifest.get("evaluation_version")
    ):
        blockers.append("evaluation_evidence_mismatch")
    bounds = evidence.get("prompt_bounds")
    if not isinstance(bounds, dict):
        return blockers + ["missing_prompt_bounds"]
    jobs = manifest["jobs"] if selected_job is None else [selected_job]
    if selected_job is None and set(bounds) != {job["job_id"] for job in jobs}:
        blockers.append("prompt_bound_job_set_mismatch")
    for job in jobs:
        proof = bounds.get(job["job_id"])
        if not isinstance(proof, dict) or not (
            _artifact_valid(proof)
            and proof.get("endpoint") == endpoint
            and proof.get("messages_sha256") == fingerprint(job["messages"])
            and type(proof.get("prompt_token_limit")) is int
            and proof.get("prompt_token_limit") == job["prompt_token_limit"]
            and type(proof.get("max_tokens")) is int
            and proof.get("max_tokens") == job["parameters"]["max_tokens"]
            and proof.get("reviewed_provider_bounds") is True
        ):
            blockers.append(f"unverified_prompt_bound:{job['job_id']}")
    return blockers


def preflight(manifest):
    """Validate offline; no credentials, persistent session, client or HTTP.

    All reservations must fit the frozen budget before beginning a run.
    This avoids silently turning a confirmatory list into a budget-selected
    subset. Costs remain conditional proxy upper bounds, not invoice estimates.
    """
    report = {
        "ready": False, "blockers": [], "job_count": 0,
        "reservation_units": None, "budget_units": None, "jobs": [],
        "unit": "nanodollar_proxy",
        "scope": "conditional_on_reviewed_provider_bounds_not_actual_invoice",
        "evidence_scope": "identity_and_integrity_checks_not_independent_verification",
    }
    try:
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object")
        with TemporaryDirectory(prefix="vib-preflight-") as directory:
            session = ExperimentSession(Path(directory) / "preview.db", manifest)
            if any(not isinstance(job.get("task_id"), str) or not job["task_id"].strip()
                   for job in manifest["jobs"]):
                raise ValueError("Every driver job needs a task_id")
            jobs = [
                {"job_id": job["job_id"], "reservation_units": session.reservation_units(job["job_id"])}
                for job in manifest["jobs"]
            ]
            report.update(
                job_count=len(jobs), jobs=jobs,
                reservation_units=sum(job["reservation_units"] for job in jobs),
                budget_units=session.summary()["budget_units"],
                manifest_sha256=fingerprint(manifest),
            )
        report["blockers"].extend(_evidence_blockers(manifest))
        if report["reservation_units"] > report["budget_units"]:
            report["blockers"].append("budget_cannot_cover_all_reservations")
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError) as error:
        report["blockers"].append(f"invalid_manifest:{type(error).__name__}")
    report["ready"] = not report["blockers"]
    return report


def run_session(manifest, session_path, client_factory, evaluators, *, approval, workers=1):
    """Execute frozen jobs with trusted, injected transport/evaluation adapters.

    ``client_factory(job_id)`` must return an endpoint-matching single-attempt
    client with its own request ledger. Evaluators are keyed by task_id and
    receive only model output. Factories and evaluators are trusted local code,
    not a sandbox or a proof that their behavior matches reviewed evidence.
    No retry, answer selection, or early stopping uses evaluation correctness.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    frozen = json.loads(json.dumps(manifest, allow_nan=False))
    preview = preflight(frozen)
    if not preview["ready"]:
        raise ValueError("Offline preflight failed; inspect blockers before execution")
    if not isinstance(approval, dict) or not (
        approval.get("approved") is True
        and approval.get("manifest_sha256") == fingerprint(frozen)
        and approval.get("budget_usd") == frozen["budget_usd"]
    ):
        raise ValueError("Explicit manifest-bound budget approval is required")
    evaluators = dict(evaluators)
    task_ids = {job["task_id"] for job in frozen["jobs"]}
    if set(evaluators) != task_ids or not all(callable(item) for item in evaluators.values()):
        raise ValueError("Exact local evaluators must be provided for all frozen task IDs")
    session = ExperimentSession(session_path, frozen)
    errors = []

    def execute(job):
        job_id = job["job_id"]
        try:
            # Recheck external artifacts just before transport or local scoring.
            if _evidence_blockers(frozen, job):
                session.stop("evidence_drift")
                raise ValueError("Evidence changed after preflight")
            state = session.job(job_id)["state"]
            if state == "planned" and not session.summary()["stop_reason"]:
                client = client_factory(job_id)
                if _evidence_blockers(frozen, job):
                    session.stop("evidence_drift")
                    raise ValueError("Evidence changed during client construction")
                session.execute(job_id, client)
            elif state not in ("received", "completed"):
                return None
            session.evaluate(job_id, evaluators[job["task_id"]], frozen["evaluation_version"])
            return None
        except Exception as error:
            session.stop("driver_failure")
            return {"job_id": job_id, "error_type": type(error).__name__}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {}
        jobs = iter(frozen["jobs"])

        def fill():
            while len(pending) < workers:
                job = next(jobs, None)
                if job is None:
                    return
                state = session.job(job["job_id"])["state"]
                if session.summary()["stop_reason"] and state not in ("received", "completed"):
                    continue
                pending[executor.submit(execute, job)] = job["job_id"]

        fill()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                error = future.result()
                if error:
                    errors.append(error)
            # Already in-flight requests settle before the executor closes.
            fill()
    rows = []
    for job in frozen["jobs"]:
        state = session.job(job["job_id"])
        rows.append({"job_id": job["job_id"], "task_id": job["task_id"],
                     "state": state["state"], "evaluation": state["evaluation"]})
    return {"preflight": preview, "summary": session.summary(),
            "jobs": rows, "errors": sorted(errors, key=lambda item: item["job_id"])}
