"""
Re-run targeted 21 MBPP Multi-Round tasks with the new gate_decisions serializer,
then report pass counts (full vs vib) to confirm we still see vib-only=2 / full-only=0
pattern that underlies Table 2 MR +9.53pp claim.

If the new workflow stats (gate_decisions field) breaks the pass pattern, we fix.
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import run_multi_benchmark as rmb

# MBPP 257 → 236 ES / 21 MR. MR set = task_ids where main summary says multi_round_count=21
# We re-derive MR tasks using main JSON's trace_results: vib_early_stopped == False OR task in MR list.
MAIN = Path("results/3dataset_synthesis_refreshed_split_or.json")
main_data = json.loads(MAIN.read_text())
traces_mbpp = main_data["trace_results"]["mbpp"]
es_ids = set()
for t in traces_mbpp:
    if t.get("full", {}).get("early_stopped") or t.get("vib", {}).get("early_stopped"):
        es_ids.add(t["task_id"])
all_ids = [t["task_id"] for t in traces_mbpp]
mr_ids = sorted([tid for tid in all_ids if tid not in es_ids])
print(f"[info] MBPP total traces {len(all_ids)}; ES={len(es_ids)}  MR={len(mr_ids)}")
assert len(mr_ids) == 21, f"need 21 MR tasks, got {len(mr_ids)}: {mr_ids[:5]}..."

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    sys.exit("Need DEEPSEEK_API_KEY env")

cfg = rmb.ExperimentConfig(
    enable_humaneval_digest=True,
    enable_mbpp_early_stop=True,
    enable_gsm8k_frozen_history=False,
    enable_math_role_weights=True,
    enable_gsm8k_reasoning_guard=False,
)
harness = rmb.MultiDatasetHarness(api_key=api_key, workers=1, experiment_config=cfg)
adapter = rmb.MBPPAdapter()
samples = adapter.load()
sb_id = {s.task_id: s for s in samples}

vib_only, full_only, both, neither = 0, 0, 0, 0
rows = []
for tid in mr_ids:
    s = sb_id[tid]
    b = harness.run_single_task_with_trace(s)
    tr = b["task_result"]
    tag = ""
    if tr.full_passed and tr.vib_passed: both += 1; tag = "BOTH"
    elif tr.vib_passed: vib_only += 1; tag = "VIB-only ★"
    elif tr.full_passed: full_only += 1; tag = "FULL-only"
    else: neither += 1; tag = "NEITHER"
    print(f"  {tid:<12}  full={int(tr.full_passed)} vib={int(tr.vib_passed)}  "
          f"f_msgs={tr.full_messages} v_msgs={tr.vib_messages}  {tag}")
    # Save gate decisions for the two vib-only for audit.
    if tag.startswith("VIB-only"):
        rows.append({
            "task_id": tid,
            "full": {"wf": b["full"]["workflow_stats"], "output": b["full"]["output"], "history": b["full"]["history"]},
            "vib":  {"wf": b["vib"]["workflow_stats"],  "output": b["vib"]["output"],  "history": b["vib"]["history"]},
        })

print(f"\nSummary: BOTH={both}  VIB-only={vib_only}  FULL-only={full_only}  NEITHER={neither}")
print(f"  Net VIB-Full = {vib_only-full_only} / {len(mr_ids)} = "
      f"{(vib_only-full_only)/len(mr_ids)*100:.2f} pp")
with open("/tmp/mr_21_rerun_gate.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"[saved] VIB-only traces → /tmp/mr_21_rerun_gate.json")
