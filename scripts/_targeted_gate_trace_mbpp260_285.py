"""Targeted rerun of MBPP/260 + MBPP/285 via run_multi_benchmark harness
to collect the new gate_decisions trace field added in workflow._vib_gatekeeper_node.

Prints:
  - for each of the two task_ids: list of gate decisions with utility / role / snip
  - passes of full vs vib (to confirm vib-only solve)

Outputs a small JSON at /tmp/vib_gate_mbpp_260_285.json for reproducibility.
Does NOT call synthesis API; only re-runs the VIB+Full history phases.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import run_multi_benchmark as rmb

TARGETS = ["MBPP/260", "MBPP/285"]
OUT = Path("/tmp/vib_gate_mbpp_260_285.json")

def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("Need DEEPSEEK_API_KEY in env (synthesis stage needs it).")

    # Use the same config as main 3dataset experiment.
    cfg = rmb.ExperimentConfig(
        enable_humaneval_digest=True,
        enable_mbpp_early_stop=True,
        enable_gsm8k_frozen_history=False,   # irrelevant for MBPP
        enable_math_role_weights=True,
        enable_gsm8k_reasoning_guard=False,
    )
    harness = rmb.MultiDatasetHarness(api_key=api_key, workers=1, experiment_config=cfg)

    adapter = rmb.MBPPAdapter()
    samples = adapter.load()
    samples_by_id = {s.task_id: s for s in samples}

    wanted = [samples_by_id[t] for t in TARGETS if t in samples_by_id]
    assert len(wanted) == len(TARGETS), f"missing {set(TARGETS) - set(samples_by_id)}"

    print(f"[info] will run targeted {[s.task_id for s in wanted]}")
    out = {}
    for s in wanted:
        bundle = harness.run_single_task_with_trace(s)
        tr = bundle["task_result"]
        print(f"\n====== {s.task_id} ======")
        print(f"  full_passed={tr.full_passed}  vib_passed={tr.vib_passed}")
        print(f"  full_msgs={tr.full_messages}  vib_msgs={tr.vib_messages}  "
              f"full_tok={tr.full_tokens}  vib_tok={tr.vib_tokens}")
        full_wf = bundle.get("full", {}).get("workflow_stats", {})
        vib_wf = bundle.get("vib", {}).get("workflow_stats", {})
        for key in ("full", "vib"):
            wf = bundle.get(key, {}).get("workflow_stats", {})
            gds = wf.get("gate_decisions", []) or []
            print(f"\n  --- {key}.gate_decisions (n={len(gds)}) ---")
            for gd in gds:
                print(f"    r{gd['round_idx']} "
                      f"{gd.get('role') or '?':<10} "
                      f"utility={gd['utility_score']:+7.3f}  "
                      f"suff={gd['sufficiency']:+.3f}  red={gd['redundancy']:+.3f}  "
                      f"β_t={gd['beta_t']:+.3f}  γ={gd['threshold']:+.3f}  "
                      f"allowed={1 if gd['is_allowed'] else 0}"
                      f"{' rescued' if gd.get('rescued') else ''}"
                      f"{' digest' if gd.get('digest') else ''}")
                if gd.get("msg_snip"):
                    print(f"       snip: {gd['msg_snip'][:120]}")
        # Collect JSON-serializable summary (ensure no dataclasses as values).
        def _wf(wf):
            return {
                "total_candidates": wf.get("total_candidates"),
                "total_approved": wf.get("total_approved"),
                "total_blocked": wf.get("total_blocked"),
                "total_digests": wf.get("total_digests"),
                "total_role_rescues": wf.get("total_role_rescues"),
                "steps_completed": wf.get("steps_completed"),
                "gate_decisions": wf.get("gate_decisions") or [],
            }
        out[s.task_id] = {
            "task_result": {
                "full_passed": tr.full_passed,
                "vib_passed": tr.vib_passed,
                "full_messages": tr.full_messages,
                "vib_messages": tr.vib_messages,
                "full_tokens": tr.full_tokens,
                "vib_tokens": tr.vib_tokens,
                "full_early_stopped": tr.full_early_stopped,
                "vib_early_stopped": tr.vib_early_stopped,
                "error_msg": tr.error_msg,
            },
            "full": {"output": bundle.get("full", {}).get("output"),
                     "history": bundle.get("full", {}).get("history"),
                     "workflow_stats": _wf(full_wf)},
            "vib":  {"output": bundle.get("vib", {}).get("output"),
                     "history": bundle.get("vib", {}).get("history"),
                     "workflow_stats": _wf(vib_wf)},
        }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[done] JSON saved → {OUT}")

if __name__ == "__main__":
    main()
