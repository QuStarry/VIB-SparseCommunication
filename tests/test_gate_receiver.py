import json
from pathlib import Path

import numpy as np
import pytest

from experiments.prepare_gate_receiver import select_streams, replay_policies, encode_streams
from tests.test_prompt_contrast_pilot import SOURCE


ROOT = Path(__file__).resolve().parents[1]
PRIORS = [ROOT / "results" / directory / "manifest.json" for directory in (
    "prompt_contrast_pilot_20261028", "prompt_contrast_replication_20261028")]


def stream():
    return {"task_id": "synthetic", "prompt": "goal",
            "history": [f"[{role}]: message {index}" for index, role in enumerate(
                ("Architect", "Coder", "Reviewer") * 2)]}


def vectors():
    unit = [1.0, 0.0, 0.0]
    return [unit] * 13


def test_sampling_excludes_both_prior_sets_without_using_labels():
    draft = json.loads(SOURCE.read_text())
    selected, exclusions = select_streams(draft, PRIORS)
    assert len(selected) == 32
    assert not {row["task_id"] for row in selected} & set(exclusions["task_ids"])
    assert len(exclusions["task_ids"]) == 72
    assert select_streams(draft, PRIORS)[0] == selected


def test_real_formula_replay_matches_naive_message_counts_per_round():
    projection = replay_policies(stream(), vectors())
    for policy in ("random_matched", "cosine_matched"):
        assert projection[policy]["round_counts"] == projection["vib"]["round_counts"]
        assert projection[policy]["indices"] == sorted(projection[policy]["indices"])
    assert projection["full"]["indices"] == list(range(6))
    assert len(projection["vib"]["decisions"]) == 6
    assert projection["vib"]["decisions"][1]["history_indices"] == [0]
    assert projection["vib"]["decisions"][4]["history_indices"] == [0, 1, 2, 3]


def test_empty_admission_budget_controls_do_not_force_messages():
    vector_rows = [[1.0, 0.0, 0.0]] + [[-1.0, 0.0, 0.0]] * 6 + [[-1.0, 0.0, 0.0]] * 6
    result = replay_policies(stream(), vector_rows)
    assert result["vib"]["indices"] == []
    assert result["random_matched"]["indices"] == []
    assert result["cosine_matched"]["indices"] == []


def test_cosine_matching_is_not_threshold_topk_and_never_uses_goal():
    result = replay_policies(stream(), vectors())
    assert result["cosine_matched"]["round_counts"] == result["vib"]["round_counts"]
    assert "selection" in result["cosine_matched"]


def test_batch_encoder_uses_bodies_and_formatted_history_separately():
    calls = []

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.tile([3.0, 4.0, 0.0], (len(texts), 1))

    result = encode_streams([stream()], Encoder())
    assert len(calls) == 1
    assert len(calls[0][0]) == 13
    assert calls[0][0][1] == "message 0"
    assert calls[0][0][7] == "[Architect]: message 0"
    assert np.linalg.norm(result["synthetic"][0]) == pytest.approx(1)


def test_invalid_or_nan_embeddings_fail_before_routing():
    with pytest.raises(ValueError):
        replay_policies(stream(), [[float("nan"), 0, 0]] * 13)
    with pytest.raises(ValueError):
        replay_policies(stream(), [[1, 0, 0]] * 12)
