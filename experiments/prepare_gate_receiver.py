"""Freeze receiver-only policies using real cached embeddings and no labels."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import random

import numpy as np

from experiments.check_prompt_contrast_readiness import _load_draft
from experiments.run_prompt_contrast_pilot import ROOT, write_json
from src.experiment_driver import file_sha256
from src.request_ledger import fingerprint
from src.vib_router import VIBSparseCommRouter


POLICIES = ("full", "vib", "random_matched", "cosine_matched")
CONFIG = {
    "beta_0": 3.2, "lambda_decay": 0.08, "temp_tau": 0.5, "gamma": -0.4,
    "adaptive": True, "alpha": 0.35, "tau_min": -1.0,
    "enable_math_role_weights": False, "enable_reasoning_guard": False,
    "enable_math_critical_guard": True,
}
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_PRIORS = [ROOT / "results" / folder / "manifest.json" for folder in (
    "prompt_contrast_pilot_20261028", "prompt_contrast_replication_20261028")]


def select_streams(draft, prior_paths):
    """Task-ID sampling independent of historical or prospective pass outcomes."""
    if len(prior_paths) != 2:
        raise ValueError("Require both previous cohort manifests")
    excluded, evidence = set(), []
    for path in map(Path, prior_paths):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids = manifest["task_ids"]
        if len(ids) != manifest["task_count"] or len(ids) != len(set(ids)):
            raise ValueError("Invalid previous task set")
        if excluded.intersection(ids):
            raise ValueError("Previous cohorts overlap")
        excluded.update(ids)
        evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    if len(excluded) != 72:
        raise ValueError("Require previous 24 plus 48 disjoint tasks")
    all_streams = {row["task_id"]: row for row in draft["candidate_streams"]}
    if len(all_streams) != 164 or not excluded <= set(all_streams):
        raise ValueError("Candidate stream or exclusion set differs")
    selected = random.Random(20261101).sample(sorted(set(all_streams) - excluded), 32)
    return [all_streams[task] for task in selected], {
        "task_ids": sorted(excluded), "manifests": evidence}


def _texts(stream):
    history = stream["history"]
    return [stream["prompt"], *(message.split(": ", 1)[1] for message in history), *history]


def encode_streams(streams, encoder):
    """Batch encode the goal, message bodies and role-formatted public histories."""
    texts = [text for stream in streams for text in _texts(stream)]
    vectors = np.asarray(encoder.encode(
        texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned invalid vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Encoder returned zero vectors")
    vectors /= norms
    return {stream["task_id"]: vectors[index * 13:(index + 1) * 13].tolist()
            for index, stream in enumerate(streams)}


def replay_policies(stream, vector_rows):
    """Apply actual VIB formula to incremental history; controls match round k."""
    vectors = np.asarray(vector_rows, dtype=np.float32)
    if (vectors.ndim != 2 or len(vectors) != 13 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError("Require 13 finite normalized embedding vectors")
    if len(stream["history"]) != 6:
        raise ValueError("Require six complete candidates")
    router = VIBSparseCommRouter(**CONFIG)
    router._embedding_cache = dict(zip(_texts(stream), vectors))
    approved, decisions = [], []
    for index, message in enumerate(stream["history"]):
        role, body = message.split(": ", 1)
        result = router.evaluate_detailed(
            candidate_msg=body, history=[stream["history"][position] for position in approved],
            goal=stream["prompt"], step_t=index // 3 + 1, role=role[1:-1])
        decisions.append({"index": index, "history_indices": list(approved), **asdict(result)})
        if result.is_allowed:
            approved.append(index)
    counts = [sum(index // 3 == round_index for index in approved) for round_index in (0, 1)]
    rng = random.Random(VIBSparseCommRouter.derive_task_seed(20261101, stream["task_id"]))
    random_indices, cosine_indices, selection = [], [], []
    for round_index, count in enumerate(counts):
        available = list(range(round_index * 3, round_index * 3 + 3))
        random_indices.extend(sorted(rng.sample(available, count)))
        for _ in range(count):
            def similarity(index):
                if not cosine_indices:
                    return 0.0
                return max(router._cosine_similarity(vectors[1 + index], vectors[7 + prior])
                           for prior in cosine_indices)
            picked = min(available, key=lambda index: (similarity(index), index))
            selection.append({"index": picked, "max_cosine": similarity(picked),
                              "previous_selected": list(cosine_indices)})
            cosine_indices.append(picked)
            available.remove(picked)
        cosine_indices.sort()
    result = {
        "full": {"indices": list(range(6)), "round_counts": [3, 3]},
        "vib": {"indices": approved, "round_counts": counts, "decisions": decisions},
        "random_matched": {"indices": random_indices, "round_counts": counts},
        "cosine_matched": {"indices": cosine_indices, "round_counts": counts, "selection": selection},
    }
    for projection in result.values():
        projection["history"] = [stream["history"][index] for index in projection["indices"]]
        projection["history_sha256"] = fingerprint(projection["history"])
        projection["history_utf8_bytes"] = sum(len(text.encode("utf-8")) for text in projection["history"])
    return result


def snapshot_evidence(path=SNAPSHOT):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError("Pinned local encoder snapshot unavailable; no download fallback")
    hashes = {str(file.relative_to(path)): file_sha256(file)
              for file in sorted(path.rglob("*")) if file.is_file()}
    for required in ("model.safetensors", "tokenizer.json", "modules.json"):
        if required not in hashes:
            raise ValueError("Encoder snapshot is incomplete")
    return {"path": str(path), "snapshot": path.name, "files_sha256": hashes,
            "scope": "real_local_encoder_file_identity_not_model_quality_proof"}


def prepare_gate_receiver(draft_path, prior_paths, output_dir, encoder=None):
    """Create separate offline artifacts; no transport or hidden scoring is loaded."""
    output, draft_path = Path(output_dir), Path(draft_path).resolve()
    if output.exists():
        raise FileExistsError("Existing frozen design cannot be overwritten")
    draft, digest = _load_draft(draft_path)
    streams, exclusions = select_streams(draft, prior_paths)
    evidence = snapshot_evidence()
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(SNAPSHOT), local_files_only=True, device="cpu")
        encoder.eval()
    metadata = {
        **evidence, "dimension": encoder.get_sentence_embedding_dimension(),
        "max_seq_length": encoder.max_seq_length,
        "text_truncation": "encoder_tokenizer_default_right_truncation_not_full_message_semantics",
        "device": "cpu", "eval_mode": not encoder.training, "dtype": "float32_l2_normalized",
        "versions": {package: version(package) for package in ("sentence-transformers", "torch", "transformers", "numpy")},
        "injected_encoder": encoder.__class__.__module__ != "sentence_transformers.SentenceTransformer",
    }
    if metadata["dimension"] != 384 or not metadata["eval_mode"]:
        raise ValueError("Real 384-dimensional evaluation-mode encoder required")
    embeddings = encode_streams(streams, encoder)
    projections = {stream["task_id"]: replay_policies(stream, embeddings[stream["task_id"]])
                   for stream in streams}
    sources = {"preparation_sha256": file_sha256(__file__),
               "router_sha256": file_sha256(ROOT / "src/vib_router.py"),
               "draft_validator_sha256": file_sha256(ROOT / "experiments/check_prompt_contrast_readiness.py")}
    design = {
        "version": "gate_receiver_design_v1", "scope": "receiver_only_shared_archival_candidates",
        "draft_path": str(draft_path), "draft_sha256": digest, "exclusions": exclusions,
        "sample_seed": 20261101, "task_count": 32, "task_ids": [row["task_id"] for row in streams],
        "streams": streams, "embeddings": embeddings, "encoder": metadata,
        "router_config": CONFIG, "same_round_history": "incremental",
        "guards": "core_router_code_diff_fix_role_complexity_enabled_no_extra_digest_or_role_minimum",
        "projections": projections, "policies": list(POLICIES),
        "budget_matching": "per_task_per_round_vib_admitted_message_count_not_token_or_invoice_matching",
        "prompt_source": {**draft["prompt_sources"]["specification_priority"],
                          "path": str(draft_path.parent / draft["prompt_sources"]["specification_priority"]["path"])},
        "evaluation": {**draft["evaluation"], "path": str(draft_path.parent / draft["evaluation"]["path"])},
        "sources": sources,
    }
    summary = {
        "task_count": 32, "candidate_messages": 192,
        "admitted_messages": {policy: sum(len(value[policy]["indices"]) for value in projections.values())
                              for policy in POLICIES},
        "vib_changed_tasks": sum(len(value["vib"]["indices"]) != 6 for value in projections.values()),
        "new_api_requests": 0, "encoder_dimension": metadata["dimension"],
        "encoder_max_seq_length": metadata["max_seq_length"],
    }
    output.mkdir()
    write_json(output / "design.json", design)
    write_json(output / "design_receipt.json", {"design_sha256": file_sha256(output / "design.json"),
                                              "summary": summary})
    return summary


def load_gate_design(path):
    path = Path(path)
    receipt = json.loads((path.parent / "design_receipt.json").read_text(encoding="utf-8"))
    if file_sha256(path) != receipt["design_sha256"]:
        raise ValueError("Frozen gate design fingerprint differs")
    design = json.loads(path.read_text(encoding="utf-8"))
    draft, digest = _load_draft(Path(design["draft_path"]))
    streams, exclusions = select_streams(draft, [row["path"] for row in design["exclusions"]["manifests"]])
    if (digest != design["draft_sha256"] or streams != design["streams"]
            or exclusions != design["exclusions"] or design["router_config"] != CONFIG
            or design["sources"]["preparation_sha256"] != file_sha256(__file__)
            or design["sources"]["router_sha256"] != file_sha256(ROOT / "src/vib_router.py")
            or design["sources"]["draft_validator_sha256"] != file_sha256(ROOT / "experiments/check_prompt_contrast_readiness.py")):
        raise ValueError("Gate design inputs or routing implementation changed")
    if snapshot_evidence(design["encoder"]["path"])["files_sha256"] != design["encoder"]["files_sha256"]:
        raise ValueError("Frozen encoder snapshot files changed")
    if any(version(package) != pinned for package, pinned in design["encoder"]["versions"].items()):
        raise ValueError("Encoder library versions changed")
    expected = {stream["task_id"]: replay_policies(stream, design["embeddings"][stream["task_id"]])
                for stream in streams}
    if expected != design["projections"]:
        raise ValueError("Embedding replay differs from frozen decisions")
    for proof in (design["prompt_source"], design["evaluation"]):
        if file_sha256(proof["path"]) != proof["sha256"]:
            raise ValueError("Prompt or scoring artifact changed")
    return design


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "results/prompt_contrast_preparation_20261028/draft.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_gate_receiver(args.draft, DEFAULT_PRIORS, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
