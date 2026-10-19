"""Table 4 (OFAT sensitivity) renderer — synthetic fixtures, code-path only.

These tests never touch the real sweep results: they assert the *rendering
contract* of ``render_table4_sensitivity`` so a half-finished sweep can never be
silently typeset into the appendix.
"""
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ANCHOR = {"beta0": 3.2, "lambda": 0.08, "gamma": -0.4}
GRID = {
    "beta0": [1.6, 2.4, 3.2, 4.0, 4.8],
    "lambda": [0.04, 0.06, 0.08, 0.10, 0.12],
    "gamma": [-0.8, -0.6, -0.4, -0.2, 0.0],
}


def load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(module_path.parent))
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _summary(delta_pp: float, msg_pct: float, tok_pct: float) -> dict:
    return {
        "total_tasks": 200,
        "full_pass_rate": 97.0,
        "vib_pass_rate": 97.0 - delta_pp,
        "pass_delta_pp": delta_pp,
        "msg_compression_pct": msg_pct,
        "token_saving_pct": tok_pct,
    }


def build_sweep(delta_pp_fn=None, msg_pct_fn=None) -> dict:
    """A complete 15-point grid; both value functions default to the anchor value."""
    rows = []
    for param, values in GRID.items():
        for value in values:
            delta_pp = 0.0 if delta_pp_fn is None else delta_pp_fn(param, value)
            msg_pct = 30.0 if msg_pct_fn is None else msg_pct_fn(param, value)
            rows.append({
                "param": param,
                "value": value,
                "name": f"sweep_{param}_{value}",
                "summary": _summary(delta_pp, msg_pct, 4.0),
                "rc": 0,
                "reused": False,
            })
    return {
        "anchor": ANCHOR,
        "limit": 200,
        "model": "deepseek-chat",
        "provider": "deepseek",
        "min_full_pass_rate": 95.0,
        "aborted": False,
        "rows": rows,
    }


def test_sensitivity_ready_accepts_complete_grid():
    module = load_module("export_latex_tables_t4_ok", "scripts/export_latex_tables.py")
    assert module._sensitivity_ready(build_sweep()) == ""


def test_sensitivity_ready_rejects_partial_grid():
    module = load_module("export_latex_tables_t4_partial", "scripts/export_latex_tables.py")
    sweep = build_sweep()
    sweep["rows"] = sweep["rows"][:14]
    assert module._sensitivity_ready(sweep)


def test_sensitivity_ready_rejects_aborted_sweep():
    module = load_module("export_latex_tables_t4_abort", "scripts/export_latex_tables.py")
    sweep = build_sweep()
    sweep["aborted"] = True
    assert module._sensitivity_ready(sweep)


def test_sensitivity_ready_rejects_point_without_summary():
    module = load_module("export_latex_tables_t4_empty", "scripts/export_latex_tables.py")
    sweep = build_sweep()
    sweep["rows"][3]["summary"] = {}
    assert module._sensitivity_ready(sweep)


def test_render_carries_label_model_and_full_grid():
    module = load_module("export_latex_tables_t4_render", "scripts/export_latex_tables.py")
    tex = module.render_table4_sensitivity(build_sweep())
    assert r"\label{tab:sensitivity}" in tex
    assert "deepseek-chat" in tex
    assert "__MODEL__" not in tex
    assert tex.count(r"\\") >= 15


def test_render_escapes_percent_and_leaves_no_placeholders():
    module = load_module("export_latex_tables_t4_escape", "scripts/export_latex_tables.py")
    tex = module.render_table4_sensitivity(build_sweep())
    assert r"\%" in tex
    assert "%%" not in tex
    assert "%.2f" not in tex and "%.1f" not in tex


def test_render_flags_every_row_when_accuracy_is_flat():
    module = load_module("export_latex_tables_t4_flat", "scripts/export_latex_tables.py")
    tex = module.render_table4_sensitivity(build_sweep())
    assert tex.count(r"\checkmark") == 15


def test_render_demotes_accuracy_outlier_only():
    module = load_module("export_latex_tables_t4_outlier", "scripts/export_latex_tables.py")

    def delta_pp_fn(param, value):
        return -2.5 if (param, value) == ("gamma", 0.0) else 0.0

    tex = module.render_table4_sensitivity(build_sweep(delta_pp_fn=delta_pp_fn))
    assert tex.count(r"\checkmark") == 14


def test_render_reports_beta0_compression_span():
    module = load_module("export_latex_tables_t4_span", "scripts/export_latex_tables.py")
    by_beta0 = {1.6: 47.0, 2.4: 37.0, 3.2: 23.0, 4.0: 13.0, 4.8: 7.0}

    def msg_pct_fn(param, value):
        return by_beta0[value] if param == "beta0" else 30.0

    tex = module.render_table4_sensitivity(build_sweep(msg_pct_fn=msg_pct_fn))
    assert r"$7.0$\%--$47.0$\%" in tex
