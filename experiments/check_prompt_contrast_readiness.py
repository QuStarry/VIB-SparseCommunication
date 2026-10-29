"""Offline endpoint metadata, conditional cost scenarios and manifest promotion."""

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse

from src.experiment_driver import file_sha256, preflight
from src.isolated_code_evaluation import load_evaluators
from src.llm_client import BUILTIN_PROVIDERS, DEFAULT_PRO_ENV, _parse_pro_env_aliases, _read_pro_env
from src.request_ledger import fingerprint


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"legacy_compact", "current_arbitration", "specification_priority"}


def endpoint_metadata(env_path, *, model="deepseek-v4-flash", credential_env="PRIVATE_API_KEY_1"):
    """Read only private catalogue metadata; do not expose credentials or mutate env."""
    if not re.fullmatch(r"PRIVATE_API_KEY(?:_\d+)?", credential_env):
        raise ValueError("Use a PRIVATE_API_KEY variable name, not a credential value")
    config = _read_pro_env(str(env_path))
    aliases = _parse_pro_env_aliases(str(env_path))
    if model in aliases:
        model_id = aliases[model]
    elif model in aliases.values():
        model_id = model
    else:
        raise ValueError("Requested model is not in the private catalogue")
    url_env = BUILTIN_PROVIDERS["private"]["base_url_env"]
    base_url = os.environ.get(url_env, "").strip() or config.get(url_env, "")
    url = urlparse(base_url)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("Require a configured HTTP endpoint without embedded credentials")
    present = bool(os.environ.get(credential_env, "").strip() or config.get(credential_env, ""))
    return {
        "provider": "private", "model": model_id, "base_url": base_url,
        "credential_env": credential_env, "credential_present": present,
        "identity_scope": "local_catalogue_resolution_not_online_verification",
        "authentication_tested": False, "official_snapshot_verified": False,
    }


def planning_scenario(draft):
    """Historical proxy scenario only; bytes and assumed overhead are NOT tokens."""
    input_rate, output_rate = Decimal("0.30"), Decimal("1.20")
    rows = []
    for variant in sorted(VARIANTS):
        jobs = [job for job in draft["jobs"] if job["prompt_variant"] == variant]
        byte_units = sum(
            sum(len(message["content"].encode("utf-8")) for message in job["messages"]) + 1024
            for job in jobs
        )
        output_cap = sum(job["parameters"]["max_tokens"] for job in jobs)
        cost = (Decimal(byte_units) * input_rate + Decimal(output_cap) * output_rate) / Decimal(1000000)
        rows.append({"variant": variant, "jobs": len(jobs), "assumed_input_units": byte_units,
                     "declared_output_cap": output_cap, "scenario_proxy_usd": str(cost)})
    output_cap = sum(row["declared_output_cap"] for row in rows)
    return {
        "historical_rate_date": "2026-10-23", "input_usd_per_million": str(input_rate),
        "output_usd_per_million": str(output_rate), "price_scope": "historical_proxy_not_current_gateway_invoice",
        "input_measure": "utf8_content_bytes_plus_assumed_1024_overhead_per_request",
        "prompt_tokens": None, "planning_only": True, "guaranteed_budget_compliance": False,
        "output_cap_only_historical_proxy_usd": str(Decimal(output_cap) * output_rate / Decimal(1000000)),
        "scenario_proxy_usd": str(sum(Decimal(row["scenario_proxy_usd"]) for row in rows)),
        "groups": rows,
    }


def _artifact_path(parent, name):
    path = (parent / name).resolve()
    try:
        path.relative_to(parent.resolve())
    except ValueError as error:
        raise ValueError("Preparation artifact must stay within its directory") from error
    return path


def _load_draft(path):
    path = Path(path)
    digest = file_sha256(path)
    recorded = json.loads((path.parent / "preparation_report.json").read_text(encoding="utf-8"))
    if recorded["draft_sha256"] != digest:
        raise ValueError("Prepared draft fingerprint changed")
    draft = json.loads(path.read_text(encoding="utf-8"))
    if draft.get("version") != "archival_prompt_contrast_draft_v1" or draft.get("status") != "not_authorized":
        raise ValueError("Require the original unapproved preparation draft")
    if file_sha256(ROOT / "experiments/prepare_prompt_contrast.py") != draft["builder_sha256"]:
        raise ValueError("Preparation builder implementation changed")
    if set(draft["prompt_sources"]) != VARIANTS:
        raise ValueError("Prompt variant set changed")
    for artifact in [*draft["prompt_sources"].values(), draft["evaluation"]]:
        if file_sha256(_artifact_path(path.parent, artifact["path"])) != artifact["sha256"]:
            raise ValueError("Prepared supporting artifact changed")
    jobs = draft["jobs"]
    ids = [job["job_id"] for job in jobs]
    tasks = {job["task_id"] for job in jobs}
    repeats = draft["repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("Invalid declared repetitions")
    combinations = {(job["task_id"], job["prompt_variant"], job["repetition"]) for job in jobs}
    expected = {(task, variant, repeat) for task in tasks for variant in VARIANTS for repeat in range(repeats)}
    if not tasks or len(tasks) != draft["task_count"] or len(set(ids)) != len(ids) or combinations != expected or len(jobs) != len(expected):
        raise ValueError("Incomplete or duplicated task/variant/repetition set")
    streams = {}
    for job in jobs:
        if job["messages_sha256"] != fingerprint(job["messages"]):
            raise ValueError("Request messages fingerprint differs")
        if job["stage"] != "code" or job["parameters"] != {"temperature": 0, "max_tokens": 1024}:
            raise ValueError("Frozen generation parameters changed")
        prior = streams.setdefault(job["task_id"], job["candidate_stream_sha256"])
        if prior != job["candidate_stream_sha256"]:
            raise ValueError("Prompt variants do not share the same candidate stream")
    if file_sha256(path) != digest:
        raise ValueError("Draft changed during validation")
    return draft, digest


def convert_draft(draft_path, endpoint, evidence):
    """Preserve all jobs; only reviewed evidence can supply prices and token bounds.

    A candidate manifest is not approval. ``run_session`` still requires a
    separate user-approved hash/budget binding before creating any client.
    """
    path = Path(draft_path)
    draft, digest = _load_draft(path)
    report = {"ready": False, "paid_authorized": False, "blockers": [], "draft_sha256": digest}
    if evidence is None:
        report["blockers"] = ["current_price_evidence_missing", "provider_prompt_bounds_missing",
                              "manifest_budget_not_approved"]
        return None, report
    public_endpoint = {key: endpoint[key] for key in ("provider", "model", "base_url")}
    if evidence.get("draft_sha256") != digest or evidence.get("endpoint") != public_endpoint:
        report["blockers"] = ["evidence_draft_or_endpoint_mismatch"]
        return None, report
    try:
        evidence = json.loads(json.dumps(evidence, allow_nan=False))
        proof = evidence["preflight"]["evaluation_evidence"]
        bundle_path = _artifact_path(path.parent, draft["evaluation"]["path"])
        if (Path(proof["path"]).resolve() != bundle_path or proof["sha256"] != draft["evaluation"]["sha256"]
                or proof["version"] != draft["evaluation"]["version"]):
            raise ValueError("Evaluation evidence must bind the prepared bundle")
        evaluators = load_evaluators(bundle_path, draft["evaluation"]["sha256"])
        if set(evaluators) != {job["task_id"] for job in draft["jobs"]}:
            raise ValueError("Evaluation task set differs from frozen jobs")
        # Reviewed evidence must not ask us to hash a credential config as proof.
        artifacts = [evidence["preflight"]["price_evidence"], *evidence["preflight"]["prompt_bounds"].values()]
        if any(Path(item["path"]).name.endswith(".env") for item in artifacts):
            raise ValueError("Credential configuration is not admissible evidence")
        fields = ("job_id", "task_id", "stage", "messages", "parameters", "prompt_variant",
                  "repetition", "candidate_stream_sha256", "messages_sha256")
        jobs = []
        for job in draft["jobs"]:
            bounds = evidence["preflight"]["prompt_bounds"][job["job_id"]]
            jobs.append({**{field: job[field] for field in fields}, "prompt_token_limit": bounds["prompt_token_limit"]})
        candidate = {
            "version": 1, **public_endpoint, "price": evidence["price"],
            "budget_usd": evidence["budget_usd"], "evaluation_version": draft["evaluation"]["version"],
            "jobs": jobs, "preflight": evidence["preflight"], "prepared_draft_sha256": digest,
            "design": draft["design"], "promotion_sha256": file_sha256(__file__),
        }
        report.update(preflight(candidate))
        report["paid_authorized"] = False
        if file_sha256(path) != digest:
            raise ValueError("Draft changed during conversion")
        return (candidate if report["ready"] else None), report
    except (KeyError, TypeError, ValueError, OSError) as error:
        report.update(ready=False, blockers=[f"invalid_reviewed_evidence:{type(error).__name__}"])
        return None, report


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_readiness(draft_path, output_dir, *, env_path=ROOT / DEFAULT_PRO_ENV,
                    model="deepseek-v4-flash", credential_env="PRIVATE_API_KEY_1", evidence_path=None):
    """Create an offline report; refuse overwrites and never execute a request."""
    draft_path, output_dir = Path(draft_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Readiness report directory already exists")
    draft, original_hash = _load_draft(draft_path)
    endpoint = endpoint_metadata(env_path, model=model, credential_env=credential_env)
    evidence_hash = file_sha256(evidence_path) if evidence_path else None
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8")) if evidence_path else None
    candidate, readiness = convert_draft(draft_path, endpoint, evidence)
    blockers = list(readiness["blockers"])
    if not endpoint["credential_present"]:
        blockers.append("selected_credential_missing")
        candidate = None
    if file_sha256(draft_path) != original_hash or (evidence_path and file_sha256(evidence_path) != evidence_hash):
        raise ValueError("Input evidence changed before output")
    scenario = planning_scenario(draft)
    report = {
        "draft_sha256": original_hash, "endpoint": endpoint, "job_count": len(draft["jobs"]),
        "candidate_ready": candidate is not None, "paid_authorized": False,
        "blockers": blockers, "scenario": scenario, "readiness": readiness,
        "input_unchanged": True, "new_model_calls": 0,
        "implementation_sha256": file_sha256(__file__),
    }
    output_dir.mkdir(parents=False)
    _write_json(output_dir / "readiness_report.json", report)
    if candidate is not None:
        _write_json(output_dir / "candidate_manifest.json", candidate)
    summary = f"""# Prompt 对照执行前就绪报告

完整请求数：{len(draft["jobs"])}。模型：`{endpoint["model"]}`。凭证仅记录变量名`{credential_env}`与是否存在，不导出或哈希其值。端点仅完成本地配置解析，未认证或验证在线模型身份。

## 条件费用场景

历史2026-10-23代理价：输入$0.30/百万、输出$1.20/百万。全部输出上限按该历史价计为${scenario["output_cap_only_historical_proxy_usd"]}，不代表实际会用满输出上限。

以“UTF-8消息内容字节数加每请求1024假设开销”作输入单位的规划场景合计${scenario["scenario_proxy_usd"]}。这不是实测Token、已证明Token上界、当前网关账单或预算合规保证，不得用于填入reviewed_provider_bounds。

## 执行状态

候选清单就绪：{report["candidate_ready"]}。付费授权：False。新模型调用：0。

阻塞项：{", ".join(blockers) or "尚需用户针对候选manifest确认预算"}。

无完整证据时不输出candidate_manifest.json；有候选清单也必须另行取得绑定其SHA256与预算的批准。历史小规模代理计价授权不自动批准本次984请求。

后续需私有网关实际价目/缓存计费依据、model对应tokenizer和chat template的上界说明，以及对新manifest的预算确认。公开模型价格或少量观测usage不能证明全部请求的真实上界。

## 一键导航

| 内容 | 入口 |
|---|---|
| 端点、分组场景与就绪详情 | [readiness_report.json](readiness_report.json) |
| 已冻结真实候选流 | [准备报告](../prompt_contrast_preparation_20261028/README.md) |
| 隔离调度与证据契约 | [使用说明](../../docs/experiment_session_driver.md) |
"""
    with (output_dir / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(summary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-path", type=Path, default=ROOT / DEFAULT_PRO_ENV)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--credential-env", default="PRIVATE_API_KEY_1")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    report = check_readiness(args.draft, args.output_dir, env_path=args.env_path, model=args.model,
                             credential_env=args.credential_env, evidence_path=args.evidence)
    print(f"jobs={report['job_count']} candidate_ready={report['candidate_ready']} paid_authorized=False")
    return 0 if report["candidate_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
