import json

import pytest

from scripts.export_prospective_report import POLICIES, load_analysis, render_report
from src.request_ledger import fingerprint


def test_offline_report_uses_data_and_labels_cost_scope():
    result = {
        "primary_comparison": {"delta_pp": 0, "ci95_pp": [0, 0], "exact_sign_p": 1,
                               "left_higher_tasks": 0, "right_higher_tasks": 0, "tied_tasks": 16},
        "actual_experiment_cost": {"proxy_usd": "0.1", "prompt_tokens": 100, "completion_tokens": 20},
        "statistics": {"groups": {policy: {"task_mean_accuracy_pct": 93.75, "passed_generations": 30}
                                  for policy in POLICIES}},
        "communication": {policy: {"messages": 80} for policy in POLICIES},
        "policy_costs": {policy: {"standalone_one_synthesis_proxy_usd": "0.01",
                                 "round2": {"prompt_tokens": 100}, "synthesis": {"prompt_tokens": 50}}
                         for policy in POLICIES},
        "truncated_by_phase": {"round1": 0, "round2": 0, "synthesis": 0},
    }
    html = render_report(result, [])
    assert "{{" not in html
    assert "93.7500%" in html
    assert "$0.01" in html
    assert "http://" not in html and "https://" not in html
    assert 'data-metric="cost"' in html
    assert "不是实际实验总支出" in html
    assert "不是总体无风险证明" in html
    assert "prefers-reduced-motion" in html
    assert "max:100" in html


def test_export_rejects_changed_analysis(tmp_path):
    payload = {"statistic": 1}
    payload["content_sha256"] = fingerprint(payload)
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(payload))
    assert load_analysis(path) == payload
    payload["statistic"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_analysis(path)
