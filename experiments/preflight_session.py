"""Offline-only preflight CLI; intentionally has no paid execution switch."""

import argparse
import json
import os
from pathlib import Path

from src.experiment_driver import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = preflight(manifest)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Exclusive creation prevents overwriting evidence or earlier reports.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Preflight failed: {type(error).__name__}\n")
    print(f"ready={report['ready']} jobs={report['job_count']} blockers={len(report['blockers'])}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
