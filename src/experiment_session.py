"""Durable, manifest-bound reservations and response recovery for experiments."""

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
import json
import hashlib
from pathlib import Path
import sqlite3
import threading

from .inference_protocol import InferenceProtocol
from .request_ledger import fingerprint


class BudgetExceeded(RuntimeError):
    """The frozen budget cannot cover another reservation."""


class SessionStopped(RuntimeError):
    """Unresolved evidence or a protocol failure blocks new paid requests."""


class ManifestMismatch(ValueError):
    """A resume or request differs from the immutable experiment contract."""


def _positive_decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid budget or price") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("Budget and prices must be finite and positive")
    return number


def _units(amount):
    # Integer nanodollars avoid cumulative float error; always round upward.
    with localcontext() as context:
        context.prec = 50
        if amount > Decimal(2**63 - 1) / 1_000_000_000:
            raise ValueError("Amount exceeds the supported integer range")
        result = int((amount * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
    if result > 2**63 - 1:
        raise ValueError("Amount exceeds the supported integer range")
    return result


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = key.lower()
            if ("api_key" in normalized or normalized in ("authorization", "password", "secret")):
                raise ValueError("Manifest must not contain credential fields")
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)


class ExperimentSession:
    """One frozen experiment shared by worker threads, with durable recovery.

    Budget guarantees are conditional on the caller's verified prompt-token
    upper bounds, provider output caps, and frozen proxy prices. Unknown usage
    retains the reservation; provider bound violations preserve actual charges.
    Reopening a session with unresolved reservations conservatively halts it.
    A live session should be shared, not reopened while requests are in flight.
    """

    def __init__(self, path, manifest):
        _reject_credentials(manifest)
        encoded = _canonical(manifest)
        self._manifest = json.loads(encoded)
        self._implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._manifest_id = fingerprint([self._manifest, self._implementation_sha256])
        self._lock = threading.RLock()
        self.path = Path(path)
        self._validate_manifest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (
                    name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'planned',
                    reserved_units INTEGER NOT NULL DEFAULT 0,
                    charged_units INTEGER,
                    valid_response INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT, response_sha256 TEXT,
                    evaluation INTEGER);
            """)
        with self._transaction() as connection:
            stored = self._metadata(connection, "manifest")
            if stored is not None and stored != encoded:
                raise ManifestMismatch("Immutable manifest differs; use a new session")
            implementation = self._metadata(connection, "implementation_sha256")
            if stored is not None and implementation != self._implementation_sha256:
                raise ManifestMismatch("Session implementation changed; do not resume old jobs")
            if stored is None:
                connection.execute("INSERT INTO metadata VALUES ('manifest', ?)", (encoded,))
                connection.execute("INSERT INTO metadata VALUES ('implementation_sha256', ?)",
                                   (self._implementation_sha256,))
                connection.execute("INSERT INTO metadata VALUES ('stop_reason', '')")
                connection.executemany(
                    "INSERT INTO jobs (job_id, spec_json) VALUES (?, ?)",
                    [(job["job_id"], _canonical(job)) for job in self._manifest["jobs"]],
                )
            rows = connection.execute("SELECT job_id, spec_json FROM jobs").fetchall()
            expected = {job["job_id"]: _canonical(job) for job in self._manifest["jobs"]}
            if {row["job_id"]: row["spec_json"] for row in rows} != expected:
                raise ManifestMismatch("Persisted job specifications differ from manifest")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE state IN ('reserved', 'unknown') LIMIT 1"
            ).fetchone():
                self._halt(connection, "unresolved_requests")

    def _validate_manifest(self):
        manifest = self._manifest
        if manifest.get("version") != 1:
            raise ValueError("Unsupported session manifest version")
        for name in ("model", "provider", "base_url", "evaluation_version"):
            if not isinstance(manifest.get(name), str) or not manifest[name].strip():
                raise ValueError(f"Missing frozen {name}")
        price = manifest["price"]
        if not price.get("version") or price.get("scope") != "proxy_not_invoice":
            raise ValueError("A versioned proxy price schedule is required")
        self._input_rate = _positive_decimal(price["input_usd_per_million"])
        self._output_rate = _positive_decimal(price["output_usd_per_million"])
        self._budget_units = _units(_positive_decimal(manifest["budget_usd"]))
        jobs = manifest["jobs"]
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("A nonempty frozen job list is required")
        ids = [job["job_id"] for job in jobs]
        if any(not isinstance(job_id, str) or not job_id for job_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate or invalid job IDs")
        for job in jobs:
            limit = job["prompt_token_limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("A verified positive prompt_token_limit is required")
            if set(job["parameters"]) != {"temperature", "max_tokens"}:
                raise ValueError("Freeze temperature and max_tokens for every job")
            protocol = InferenceProtocol(**job["parameters"])
            if protocol.temperature is None or protocol.max_tokens is None:
                raise ValueError("Job parameters cannot be unspecified")
            messages = job["messages"]
            if not isinstance(messages, list) or not messages or any(
                set(message) != {"role", "content"} or
                message["role"] not in ("system", "user", "assistant") or
                not isinstance(message["content"], str) for message in messages
            ):
                raise ValueError("Freeze complete text messages, not evaluation material")
            if "reference" in job or "test_cases" in job or "passed" in job:
                raise ValueError("Evaluation material must stay outside inference jobs")
            self._price_units(limit, protocol.max_tokens)
        self._jobs = {job["job_id"]: job for job in jobs}

    def _price_units(self, prompt_tokens, completion_tokens):
        with localcontext() as context:
            context.prec = 50
            amount = (prompt_tokens * self._input_rate + completion_tokens * self._output_rate) / 1_000_000
            return _units(amount)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous = FULL")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self):
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _metadata(connection, name):
        row = connection.execute("SELECT value FROM metadata WHERE name = ?", (name,)).fetchone()
        return row["value"] if row else None

    def _halt(self, connection, reason):
        if not self._metadata(connection, "stop_reason"):
            connection.execute("UPDATE metadata SET value = ? WHERE name = 'stop_reason'", (reason,))

    def stop(self, reason):
        with self._transaction() as connection:
            self._halt(connection, reason)

    @staticmethod
    def _totals(connection):
        row = connection.execute("""
            SELECT COALESCE(SUM(charged_units), 0) AS spent,
                   COALESCE(SUM(CASE WHEN state IN ('reserved', 'unknown')
                        AND charged_units IS NULL THEN reserved_units ELSE 0 END), 0) AS held
            FROM jobs
        """).fetchone()
        return row["spent"], row["held"]

    def reservation_units(self, job_id):
        """Quote one frozen request's proxy upper bound in integer nanodollars."""
        spec = self._jobs[job_id]
        return self._price_units(spec["prompt_token_limit"], spec["parameters"]["max_tokens"])

    def claim(self, job_id):
        """Atomically reserve before sending; False means reuse a valid response."""
        reservation = self.reservation_units(job_id)
        error = None
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row["state"] in ("received", "completed") and row["valid_response"]:
                return False
            reason = self._metadata(connection, "stop_reason")
            if reason or row["state"] != "planned":
                error = SessionStopped(f"Session stopped or job is {row['state']}: {reason or 'reserved'}")
            else:
                spent, held = self._totals(connection)
                if spent + held + reservation > self._budget_units:
                    self._halt(connection, "budget_limit")
                    error = BudgetExceeded("Shared budget cannot cover this reservation")
                else:
                    connection.execute(
                        "UPDATE jobs SET state = 'reserved', reserved_units = ? WHERE job_id = ?",
                        (reservation, job_id),
                    )
        if error:
            raise error
        return True

    def job(self, job_id):
        """Detached job state; verify stored response integrity before reuse."""
        with self._connection() as connection:
            row = dict(connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())
        row["spec"] = json.loads(row.pop("spec_json"))
        encoded = row.pop("response_json")
        try:
            row["response"] = json.loads(encoded) if encoded else None
        except ValueError:
            self.stop("response_integrity")
            raise SessionStopped("Persisted response is not valid JSON") from None
        if row["response"] is not None and fingerprint(row["response"]) != row["response_sha256"]:
            self.stop("response_integrity")
            raise SessionStopped("Persisted response fingerprint differs")
        return row

    def _receive(self, job_id, response, failure=None):
        response = {**response, "manifest_sha256": self._manifest_id, "experiment_job_id": job_id}
        spec = self._jobs[job_id]
        prompt = response.get("prompt_tokens")
        completion = response.get("completion_tokens")
        known = response.get("usage_status") == "known" and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (prompt, completion)
        )
        charge = self._price_units(prompt, completion) if known else None
        reason = failure
        if not known:
            reason = reason or "unknown_usage"
        elif prompt > spec["prompt_token_limit"] or completion > spec["parameters"]["max_tokens"]:
            reason = "provider_bound_violation"
        elif response.get("served_model") != self._manifest["model"]:
            reason = reason or "served_model_drift"
        elif response.get("finish_reason") != "stop" or not response.get("content") or not response.get("response_id"):
            reason = reason or "invalid_response"
        encoded = _canonical(response)
        with self._transaction() as connection:
            row = connection.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row["state"] != "reserved":
                raise SessionStopped("No active reservation for this response")
            connection.execute("""
                UPDATE jobs SET state = ?, charged_units = ?, valid_response = ?,
                    response_json = ?, response_sha256 = ? WHERE job_id = ?
            """, ("unknown" if reason else "received", charge, int(reason is None),
                  encoded, fingerprint(response), job_id))
            if reason:
                self._halt(connection, reason)
        if reason:
            raise SessionStopped(f"Request evidence requires audit: {reason}") from None
        return response

    def execute(self, job_id, client):
        """Use one real client attempt; retries require separately frozen job IDs."""
        for name in ("model", "provider", "base_url"):
            if getattr(client, name, None) != self._manifest[name]:
                raise ManifestMismatch(f"Client {name} differs from frozen endpoint")
        if not self.claim(job_id):
            return self.job(job_id)["response"]
        spec = self._jobs[job_id]
        operation_id = fingerprint([self._manifest_id, job_id])
        try:
            result = client.chat_single_attempt(
                spec["messages"], **spec["parameters"], operation_id=operation_id,
                stage=spec.get("stage"),
            )
        except Exception as error:
            records = [record for record in client.get_request_records()
                       if record.get("operation_id") == operation_id]
            evidence = records[-1] if records else {}
            usage = evidence.get("usage") or {}
            response = {
                "usage_status": evidence.get("usage_status", "unknown"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "served_model": evidence.get("served_model"),
                "response_id": evidence.get("response_id"),
                "request_evidence": evidence, "error_type": type(error).__name__,
            }
            return self._receive(job_id, response, failure="request_failure")
        return self._receive(job_id, result)

    def evaluate(self, job_id, evaluator, evaluation_version):
        """Resume local evaluation only; never regenerate an already received output."""
        if evaluation_version != self._manifest["evaluation_version"]:
            raise ManifestMismatch("Evaluation version differs from the frozen protocol")
        row = self.job(job_id)
        if row["state"] == "completed":
            return bool(row["evaluation"])
        if row["state"] != "received" or not row["valid_response"]:
            raise SessionStopped("No valid received output to evaluate")
        try:
            passed = evaluator(row["response"]["content"])
            if not isinstance(passed, bool):
                raise ValueError("Evaluator must return a boolean")
        except Exception as error:
            self.stop("evaluation_failure")
            raise SessionStopped("Local evaluation failed; output retained") from error
        with self._transaction() as connection:
            current = connection.execute("SELECT evaluation FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if current["evaluation"] is not None and bool(current["evaluation"]) != passed:
                self._halt(connection, "inconsistent_evaluation")
                inconsistent = True
            else:
                connection.execute(
                    "UPDATE jobs SET state = 'completed', evaluation = ? WHERE job_id = ?",
                    (int(passed), job_id),
                )
                inconsistent = False
        if inconsistent:
            raise SessionStopped("Concurrent local evaluations disagree")
        return passed

    def summary(self):
        """Known charges and unresolved holds are separate; prices remain proxies."""
        with self._connection() as connection:
            spent, held = self._totals(connection)
            counts = dict(connection.execute(
                "SELECT state, COUNT(*) FROM jobs GROUP BY state"
            ).fetchall())
            unknown_usage = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('reserved', 'unknown') AND charged_units IS NULL"
            ).fetchone()[0]
            return {
                "budget_units": self._budget_units, "known_spent_units": spent,
                "held_units": held, "committed_units": spent + held,
                "reserved_requests": counts.get("reserved", 0),
                "unknown_requests": counts.get("unknown", 0),
                "received_requests": counts.get("received", 0),
                "completed_requests": counts.get("completed", 0),
                "planned_requests": counts.get("planned", 0),
                "unknown_usage_requests": unknown_usage, "usage_complete": unknown_usage == 0,
                "run_complete": counts.get("completed", 0) == len(self._jobs),
                "manifest_sha256": self._manifest_id,
                "stop_reason": self._metadata(connection, "stop_reason") or None,
                "unit": "nanodollar_proxy", "price_version": self._manifest["price"]["version"],
                "scope": "conditional_on_verified_bounds_not_actual_invoice",
            }
