import argparse
import gzip
import json
from pathlib import Path

from run_multi_benchmark import MultiDatasetHarness, TaskSample
from llm_client import resolve_deepseek_api_key


DEFAULT_TASK_IDS = [
    "HumanEval/54",
    "HumanEval/65",
    "HumanEval/130",
    "HumanEval/135",
    "HumanEval/144",
]


def load_target_samples(task_ids: list[str]) -> list[TaskSample]:
    dataset_path = Path(__file__).with_name("HumanEval.jsonl.gz")
    samples: list[TaskSample] = []
    wanted = set(task_ids)

    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["task_id"] not in wanted:
                continue
            samples.append(
                TaskSample(
                    task_id=obj["task_id"],
                    dataset_name="HumanEval",
                    prompt=obj["prompt"],
                    test_cases=obj["test"],
                    entry_point=obj["entry_point"],
                    metadata={}
                )
            )

    samples.sort(key=lambda sample: sample.task_id)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run targeted HumanEval tasks")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=DEFAULT_TASK_IDS,
        help="Target HumanEval task ids"
    )
    parser.add_argument(
        "--output",
        default="results/humaneval_full_only_targeted_after_codegen_fix.json",
        help="Output JSON path"
    )
    args = parser.parse_args()

    api_key = resolve_deepseek_api_key()
    if not api_key:
        raise SystemExit("missing DEEPSEEK_API_KEY")

    samples = load_target_samples(args.task_ids)
    harness = MultiDatasetHarness(api_key=api_key, workers=1)
    records = []

    for sample in samples:
        bundle = harness.run_single_task_with_trace(sample)
        task = bundle["task_result"]
        print(sample.task_id, task.full_passed, task.vib_passed, task.full_messages, task.vib_messages)
        records.append(
            {
                "task_id": sample.task_id,
                "full_passed": task.full_passed,
                "vib_passed": task.vib_passed,
                "full_messages": task.full_messages,
                "vib_messages": task.vib_messages,
                "full_output": bundle["full"]["output"] if bundle["full"] else None,
                "vib_output": bundle["vib"]["output"] if bundle["vib"] else None,
                "full_history": bundle["full"]["history"] if bundle["full"] else None,
                "vib_history": bundle["vib"]["history"] if bundle["vib"] else None,
                "full_workflow_stats": bundle["full"]["workflow_stats"] if bundle["full"] else None,
                "vib_workflow_stats": bundle["vib"]["workflow_stats"] if bundle["vib"] else None,
            }
        )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
