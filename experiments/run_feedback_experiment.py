"""Stage-aware single-attempt execution and local routing of fresh candidates."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np

from experiments.check_prompt_contrast_readiness import endpoint_metadata
from experiments.feedback_protocol import ROLES, freeze_feedback_protocol, stage_manifest, validate_protocol
from experiments.prepare_gate_receiver import SNAPSHOT
from experiments.run_prompt_contrast_pilot import ROOT, configure_live_client, response_path, write_json
from src.experiment_driver import file_sha256
from src.exploratory_prompt_session import execute_exploratory, validate_exploratory_generation
from src.llm_client import DEFAULT_PRO_ENV
from src.request_ledger import fingerprint
from src.vib_router import VIBSparseCommRouter


class StageClient:
    """Adapt the frozen legacy executor without changing it or its global state."""

    def __init__(self, client, stage):
        self.wrapped = client
        self.stage = stage

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    @property
    def _client(self):
        return self.wrapped._client

    @_client.setter
    def _client(self, value):
        self.wrapped._client = value

    @property
    def request_context(self):
        return self.wrapped.request_context

    @request_context.setter
    def request_context(self, value):
        self.wrapped.request_context = value

    def chat_single_attempt(self, messages, *, stage, **kwargs):
        return self.wrapped.chat_single_attempt(messages, stage=self.stage, **kwargs)


def validate_phase(protocol, manifest):
    validate_protocol(protocol)
    if manifest != stage_manifest(protocol, manifest["phase"], manifest["parents"], manifest["routes"]):
        raise ValueError("Frozen feedback phase changed")
    for proof in manifest["parents"].values():
        if file_sha256(proof["path"]) != proof["sha256"]:
            raise ValueError("Parent phase artifact changed")
    if manifest["phase"] != "round1":
        previous = "round1" if manifest["phase"] == "round2" else "round2"
        route_key = "routes_" + previous
        route_path = manifest["parents"][route_key]["path"]
        if json.loads(Path(route_path).read_text(encoding="utf-8")) != manifest["routes"]:
            raise ValueError("Phase history differs from its parent route")
        parent_manifest_path = Path(manifest["parents"][previous + "_manifest"]["path"])
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        parent_report = json.loads(Path(manifest["parents"][previous]["path"]).read_text(encoding="utf-8"))
        validate_phase_report(protocol, parent_manifest, parent_manifest_path.parent, parent_report)
        if not parent_report["summary"]["run_complete"] or not parent_report["summary"]["usage_complete"]:
            raise ValueError("Incomplete parent cannot authorize downstream requests")


def validate_phase_report(protocol, manifest, output, report):
    validate_phase(protocol, manifest)
    records = validate_exploratory_generation(manifest, output, report)
    expected_stage = "code" if manifest["phase"] == "synthesis" else "agent"
    if any(record.get("stage") != expected_stage for record in records):
        raise ValueError("Request ledger stage differs from frozen phase")
    return records


def execute_phase(protocol, manifest, output, client_factory):
    stage = "code" if manifest["phase"] == "synthesis" else "agent"
    factory = None if client_factory is None else lambda path: StageClient(client_factory(path), stage)
    report = execute_exploratory(manifest, output, factory, lambda value: validate_phase(protocol, value))
    if "summary" in report:
        validate_phase_report(protocol, manifest, output, report)
    return report


def route_round(protocol, candidates, previous, step, encoder):
    """Gate own candidates incrementally; controls match VIB's current round k."""
    texts = []
    for task in protocol["tasks"]:
        texts.append(task["prompt"])
        for policy in protocol["policies"]:
            bodies = candidates[task["task_id"]][policy]
            if len(bodies) != 3 or any(not isinstance(body, str) or not body.strip() for body in bodies):
                raise ValueError("Require three nonempty received candidates per arm")
            texts.extend(bodies)
            texts.extend(f"[{role}]: {body}" for role, body in zip(ROLES, bodies))
            if previous:
                texts.extend(previous["histories"][task["task_id"]][policy]["history"])
    texts = list(dict.fromkeys(texts))
    vectors = np.asarray(encoder.encode(texts, convert_to_numpy=True, batch_size=32,
                                       show_progress_bar=False), dtype=np.float32)
    if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid real encoder vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero encoder vectors")
    if isinstance(encoder, StoredEncoder):
        if not np.allclose(norms, 1, atol=1e-5):
            raise ValueError("Cached encoder vectors are not normalized")
    else:
        vectors /= norms
    cache = dict(zip(texts, vectors))
    histories = {}
    for task in protocol["tasks"]:
        task_id = task["task_id"]
        arms, count = {}, None
        for policy in ("vib", "full", "random_matched", "cosine_matched"):
            prior = previous["histories"][task_id][policy] if previous else {"history": [], "round_counts": []}
            history = list(prior["history"])
            bodies = candidates[task_id][policy]
            formatted = [f"[{role}]: {body}" for role, body in zip(ROLES, bodies)]
            router = VIBSparseCommRouter(**protocol["router_config"])
            router._embedding_cache = cache
            decisions = []
            if policy == "vib":
                indices = []
                for index, (role, body) in enumerate(zip(ROLES, bodies)):
                    result = router.evaluate_detailed(candidate_msg=body, history=history,
                                                      goal=task["prompt"], step_t=step, role=role)
                    decisions.append({"index": index, "history_sha256": fingerprint(history), **asdict(result)})
                    if result.is_allowed:
                        indices.append(index)
                        history.append(formatted[index])
                count = len(indices)
            elif policy == "full":
                indices = list(range(3))
                history.extend(formatted)
            elif policy == "random_matched":
                seed = VIBSparseCommRouter.derive_task_seed(20261103 + step, task_id)
                indices = sorted(random.Random(seed).sample(range(3), count))
                history.extend(formatted[index] for index in indices)
            else:
                available, selected_history, picked = list(range(3)), list(history), []
                for _ in range(count):
                    def similarity(index):
                        return max((router._cosine_similarity(cache[bodies[index]], cache[text])
                                    for text in selected_history), default=0.0)
                    index = min(available, key=lambda item: (similarity(item), item))
                    decisions.append({"index": index, "max_cosine": similarity(index)})
                    picked.append(index)
                    selected_history.append(formatted[index])
                    available.remove(index)
                indices = sorted(picked)
                history.extend(formatted[index] for index in indices)
            arms[policy] = {
                "history": history, "history_sha256": fingerprint(history),
                "round_counts": [*prior["round_counts"], len(indices)],
                "current_indices": indices, "decisions": decisions,
                "history_utf8_bytes": sum(len(text.encode("utf-8")) for text in history),
            }
        histories[task_id] = arms
    return {"step": step, "histories": histories, "embedding_texts": texts,
            "embedding_vectors": vectors.tolist(), "candidates_sha256": fingerprint(candidates),
            "previous_sha256": fingerprint(previous), "protocol_sha256": fingerprint(protocol)}


def received_candidates(protocol, manifest, output):
    indexed = {}
    for job in manifest["jobs"]:
        raw = json.loads(response_path(output, job).read_text(encoding="utf-8"))
        indexed[(job["task_id"], job["prompt_variant"], job["agent_role"])] = raw["content"]
    return {
        task: {policy: [indexed[(task, "shared" if manifest["phase"] == "round1" else policy, role)]
                       for role in ROLES] for policy in protocol["policies"]}
        for task in protocol["task_ids"]
    }


def proof(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def immutable_json(path, value):
    if Path(path).exists():
        if json.loads(Path(path).read_text(encoding="utf-8")) != value:
            raise ValueError("Existing frozen artifact differs")
    else:
        write_json(path, value)


class StoredEncoder:
    def __init__(self, route):
        self.route = route

    def encode(self, texts, **kwargs):
        if texts != self.route["embedding_texts"]:
            raise ValueError("Cached route text inputs changed")
        return self.route["embedding_vectors"]


def run_feedback(output_dir, client_factory=None):
    """Run immutable phases; stop all downstream generation on incomplete usage."""
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    lock = output / ".pipeline.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        protocol_path = output / "protocol.json"
        protocol = (json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists()
                    else freeze_feedback_protocol(endpoint_metadata(ROOT / DEFAULT_PRO_ENV)))
        validate_protocol(protocol)
        immutable_json(protocol_path, protocol)
        parents, routes, encoder = {}, None, None
        summaries = {}
        for step, phase in enumerate(("round1", "round2", "synthesis"), 1):
            manifest = stage_manifest(protocol, phase, parents, routes)
            phase_output = output / phase
            report = execute_phase(protocol, manifest, phase_output, client_factory)
            if "summary" not in report:
                return {"prepared": True, "phase": phase, "planned_proxy_scenario_usd":
                        manifest["planned_proxy_scenario_usd"]}
            summaries[phase] = report["summary"]
            if not report["summary"]["run_complete"] or not report["summary"]["usage_complete"]:
                return {"complete": False, "summaries": summaries}
            parents = {**parents, phase: proof(phase_output / "generation_report.json"),
                       phase + "_manifest": proof(phase_output / "manifest.json")}
            if phase == "synthesis":
                break
            candidates = received_candidates(protocol, manifest, phase_output)
            route_path = output / f"routes_round{step}.json"
            if route_path.exists():
                stored = json.loads(route_path.read_text(encoding="utf-8"))
                recomputed = route_round(protocol, candidates, routes, step, StoredEncoder(stored))
                if (recomputed["histories"] != stored["histories"]
                        or recomputed["candidates_sha256"] != stored["candidates_sha256"]
                        or stored["previous_sha256"] != fingerprint(routes)
                        or stored["protocol_sha256"] != fingerprint(protocol)):
                    raise ValueError("Cached local routing differs")
                routes = stored
            else:
                if encoder is None:
                    from sentence_transformers import SentenceTransformer
                    encoder = SentenceTransformer(str(SNAPSHOT), local_files_only=True, device="cpu")
                    encoder.eval()
                    if encoder.get_sentence_embedding_dimension() != 384 or encoder.max_seq_length != 256:
                        raise ValueError("Pinned encoder configuration differs")
                routes = route_round(protocol, candidates, routes, step, encoder)
                write_json(route_path, routes)
            parents = {**parents, f"routes_round{step}": proof(route_path)}
        result = {"complete": True, "summaries": summaries,
                  "protocol_sha256": fingerprint(protocol), "artifacts": parents}
        immutable_json(output / "pipeline_report.json", result)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    endpoint = endpoint_metadata(ROOT / DEFAULT_PRO_ENV)
    result = run_feedback(args.output_dir, (lambda path: configure_live_client(endpoint, path))
                          if args.execute else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("complete") or result.get("prepared") else 2


if __name__ == "__main__":
    raise SystemExit(main())
