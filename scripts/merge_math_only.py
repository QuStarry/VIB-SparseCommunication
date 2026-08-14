#!/usr/bin/env python3
"""
Merge MATH-only result JSONs into 3-dataset vanilla/main JSONs.

Strategy:
- ORIGINAL 3-dataset JSONs are NEVER modified (read-only, byte-level unchanged)
- Deep-copy original → copy math summary + detailed_results from MATH-only JSON → write NEW *_with_math_merged.json file
- Verify: old (humaneval, mbpp, gsm8k) blocks byte-by-byte identical to originals
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


PROJ = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "base_vanilla": PROJ / "results/vanilla_baseline_3dataset.json",
    "base_main":    PROJ / "results/3dataset_synthesis_refreshed_split_or.json",
    "math_vanilla": PROJ / "results/math_vanilla_5000.json",
    "math_main":    PROJ / "results/math_main_5000.json",
    "out_vanilla":  PROJ / "results/vanilla_with_math_merged.json",
    "out_main":     PROJ / "results/main_with_math_merged.json",
}


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def save_json(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ensure_math_present(math_src: dict, label: str) -> None:
    if "summary" not in math_src or "math" not in math_src.get("summary", {}):
        raise RuntimeError(f"{label}: missing summary.math")
    if "detailed_results" not in math_src or "math" not in math_src.get("detailed_results", {}):
        raise RuntimeError(f"{label}: missing detailed_results.math")
    s = math_src["summary"]["math"]
    if s.get("total_tasks", 0) < 1:
        raise RuntimeError(f"{label}: summary.math.total_tasks invalid: {s}")


def merge_one(base: dict, math_only: dict, *, label: str) -> dict:
    """Return a NEW dict = deepcopy(base) + deepcopy(math_only.{summary,detailed,trace}.math)."""
    merged = copy.deepcopy(base)

    for block in ["summary", "detailed_results", "trace_results"]:
        if block in math_only and "math" in math_only[block]:
            if block not in merged:
                merged[block] = {}
            # deep copy to ensure no aliasing
            merged[block]["math"] = copy.deepcopy(math_only[block]["math"])

    # Also merge experiment_config.meta from math if present (and keep base)
    if "experiment_config" in math_only:
        merged.setdefault("experiment_config", {})
        merged["experiment_config"].setdefault("merged_math", {})
        merged["experiment_config"]["merged_math"].update(copy.deepcopy(math_only["experiment_config"]))
    if "meta" in math_only:
        merged.setdefault("meta", {})
        merged["meta"].setdefault("merged_math", {})
        merged["meta"]["merged_math"].update(copy.deepcopy(math_only["meta"]))

    return merged


def _block_hash(d: dict, keys):
    """Stable hash for specific sub-keys: for humaneval/mbpp/gsm8k immutability check."""
    sub = {k: d.get(k, {}) for k in keys} if isinstance(keys, (list, tuple)) else {keys[0]: d.get(keys[0], {})}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def verify_immutable(base: dict, merged: dict, datasets=("humaneval", "mbpp", "gsm8k")):
    """Assert that the three dataset blocks are byte-identical between base and merged."""
    for block in ("summary", "detailed_results", "trace_results"):
        b = base.get(block, {})
        m = merged.get(block, {})
        for ds in datasets:
            if ds not in b:
                continue
            bh = _block_hash(b, [ds])
            mh = _block_hash(m, [ds])
            if bh != mh:
                raise AssertionError(
                    f"IMMUTABILITY VIOLATION: block={block} ds={ds} "
                    f"base_hash={bh} merged_hash={mh}"
                )
    print(f"  ✅ 3-dataset immutable check PASS ({len(datasets)} datasets × 3 blocks).")


def _summarize_math(merged: dict, label: str):
    s = merged.get("summary", {}).get("math", {})
    n = len(merged.get("detailed_results", {}).get("math", []))
    print(f"  · {label}: math N={n} summary.total_tasks={s.get('total_tasks')} "
          f"vanilla/full={s.get('full_pass_rate')} vib={s.get('vib_pass_rate')} "
          f"msg%={s.get('msg_compression_pct')} tok%={s.get('token_saving_pct')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-vanilla", type=Path, default=DEFAULTS["base_vanilla"])
    ap.add_argument("--base-main",    type=Path, default=DEFAULTS["base_main"])
    ap.add_argument("--math-vanilla", type=Path, default=DEFAULTS["math_vanilla"])
    ap.add_argument("--math-main",    type=Path, default=DEFAULTS["math_main"])
    ap.add_argument("--out-vanilla",  type=Path, default=DEFAULTS["out_vanilla"])
    ap.add_argument("--out-main",     type=Path, default=DEFAULTS["out_main"])
    ap.add_argument("--no-backup",    action="store_true", help="don't backup originals")
    args = ap.parse_args()

    for k, p in {
        "base_vanilla": args.base_vanilla,
        "base_main":    args.base_main,
        "math_vanilla": args.math_vanilla,
        "math_main":    args.math_main,
    }.items():
        if not p.exists():
            print(f"❌ MISSING {k}: {p}")
            sys.exit(2)

    print("🔒 Step 1: read + hash originals")
    print(f"  · base_vanilla = {args.base_vanilla} ({sha256_file(args.base_vanilla)})")
    print(f"  · base_main    = {args.base_main}    ({sha256_file(args.base_main)})")

    # --- 100% safe: NEVER write to originals. Backup to results/.backup/ first.
    if not args.no_backup:
        import shutil, time
        bkdir = PROJ / "results" / ".backup"
        bkdir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (args.base_vanilla, args.base_main):
            dst = bkdir / f"{p.stem}.orig_{ts}.json"
            shutil.copy2(p, dst)
            print(f"  🛡️  backup: {dst.name}  (sha256={sha256_file(dst)})")

    print()
    print("🔍 Step 2: verify math-only JSONs are valid")
    bv = load_json(args.base_vanilla)
    bm = load_json(args.base_main)
    mv = load_json(args.math_vanilla)
    mm = load_json(args.math_main)
    _ensure_math_present(mv, str(args.math_vanilla.name))
    _ensure_math_present(mm, str(args.math_main.name))
    print("  ✅ both math-only JSONs schema OK")

    print()
    print("🧬 Step 3: deep-copy merge (base untouched, new merged dicts)")
    out_v = merge_one(bv, mv, label="vanilla+math")
    out_m = merge_one(bm, mm, label="main+math")

    print()
    print("🧪 Step 4: immutability assertions (humaneval/mbpp/gsm8k must be identical)")
    verify_immutable(bv, out_v)
    verify_immutable(bm, out_m)

    print()
    print("📋 Step 5: math summary preview")
    _summarize_math(out_v, "vanilla merged")
    _summarize_math(out_m, "main    merged")

    print()
    print("💾 Step 6: write new merged JSONs (originals untouched)")
    save_json(args.out_vanilla, out_v)
    save_json(args.out_main, out_m)
    print(f"  · {args.out_vanilla} ({args.out_vanilla.stat().st_size} bytes sha256={sha256_file(args.out_vanilla)})")
    print(f"  · {args.out_main}    ({args.out_main.stat().st_size} bytes sha256={sha256_file(args.out_main)})")

    print()
    print("✅ Merge complete.")
    print("   Next:  python3 scripts/export_latex_tables.py --vanilla", args.out_vanilla, "--main", args.out_main)


if __name__ == "__main__":
    main()
