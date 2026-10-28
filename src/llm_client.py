"""
LLM Client for VIB-SparseComm Multi-Agent System.

Abstracts over multiple OpenAI-compatible providers (DeepSeek public API,
a private OpenAI-compatible gateway, ...) driven by a single
``configs/pro.env`` lookup, so every experimental entry-point can switch
providers / aliases by changing one environment variable or CLI flag.
"""

from __future__ import annotations

import os
import re
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple

try:
    from .inference_protocol import InferenceProtocol
    from .request_ledger import RequestLedger, LedgerError, fingerprint
except ImportError:
    from inference_protocol import InferenceProtocol
    from request_ledger import RequestLedger, LedgerError, fingerprint


# ---------------------------------------------------------------------------
# Provider registry. Add backends here.
# ---------------------------------------------------------------------------

DEFAULT_PRO_ENV = "configs/pro.env"

# Canonical provider-id -> (env-var for api-key, default base-url, human label)
BUILTIN_PROVIDERS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "label": "DeepSeek Public API",
    },
    "private": {
        "api_key_env": "PRIVATE_API_KEY",
        "base_url_env": "PRIVATE_API_URL",
        "label": "Private OpenAI-compatible Gateway",
    },
}

# Model-alias dictionary used when resolving logical names.
# Order matters: first match wins (allows user to override via pro.env).
_LEGACY_ALIASES: Dict[str, Tuple[str, str]] = {
    # (provider, actual-model-id)
    "deepseek-chat": ("deepseek", "deepseek-chat"),
    "deepseek-reasoner": ("deepseek", "deepseek-reasoner"),
}

# Preferred target suffix / release tags: if the user calls
# ``build_client("deepseek-v4-flash")`` and the TSV block only lists the
# tagged forms (preview / 0817 / 正式版), pick the stable release tag
# instead of the preview. This list is checked in order (first match wins).
_PREFERRED_TAGS: Tuple[str, ...] = (
    "0817",
    "stable",
    "正式版",
    "正式",
    "release",
)

# Short-name to target-model override: only used when the TSV block does NOT
# contain an alias for the short name (user can fully override by adding one
# to the pro.env TSV section).
_PREFERRED_MODEL_REDIRECTS: Dict[str, str] = {
    "deepseek-v4-flash": "deepseek-v4-flash-0817",
    "deepseek-v4-pro": "deepseek-v4-pro-0817",
}


# ---------------------------------------------------------------------------
# pro.env parser. Keeps backwards compat with the previous hand-written
# ``resolve_deepseek_api_key`` helper.
# ---------------------------------------------------------------------------

def _read_pro_env(env_path: str = DEFAULT_PRO_ENV) -> Dict[str, str]:
    """Return the raw KEY=VALUE map from a pro.env file (if present)."""
    out: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return out
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    out[key] = value
    except OSError:
        pass
    return out


def _parse_pro_env_aliases(env_path: str = DEFAULT_PRO_ENV) -> Dict[str, str]:
    """Parse the TSV-style model-alias section of pro.env.

    Lines are expected to look like::

        deepseek-v4-flash (正式版)    deepseek-v4-flash-0817

    i.e. ``<display-name (comment)?>\\t<model-id>``. The left-hand side
    is stripped of parenthetical comments and any surrounding whitespace
    before being used as the alias key.
    """
    aliases: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return aliases
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if "\t" not in line:
                    continue
                lhs, rhs = line.split("\t", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                if not lhs or not rhs:
                    continue
                if lhs in ("模型名称", "Model id") or rhs in ("模型名称", "Model id"):
                    continue
                # Drop any parenthetical qualifier, e.g. "(正式版)".
                lhs_clean = re.sub(r"\s*\([^)]*\)\s*$", "", lhs).strip()
                has_qualifier = (lhs_clean != lhs)
                if lhs_clean:
                    # Clean (no-qualifier) rows win: if the row has a qualifier
                    # (preview / 正式版) we use setdefault so a prior clean row
                    # (e.g. "deepseek-v4-flash" → deepseek-v4-flash-0817) is
                    # never shadowed by a later bracketed form.
                    if has_qualifier:
                        aliases.setdefault(lhs_clean, rhs)
                    else:
                        aliases[lhs_clean] = rhs
                # Also register the display verbatim form (for copy-paste).
                aliases[lhs] = rhs
    except OSError:
        pass
    return aliases


@dataclass
class ResolvedModel:
    """Fully-qualified LLM endpoint description."""
    provider: str
    model_id: str
    base_url: str
    api_key: str = field(repr=False)
    label: str

    def describe(self) -> str:
        return f"[{self.provider}] {self.label} :: {self.model_id} @ {self.base_url}"


def _resolve_provider(
    provider_arg: Optional[str],
    logical_model: str,
    env_path: str = DEFAULT_PRO_ENV,
) -> ResolvedModel:
    """Resolve (provider, model) to concrete endpoint + credentials.

    Resolution order:

    1. Explicit ``--provider`` flag.
    2. ``VIB_LLM_PROVIDER`` environment variable.
    3. Model-name heuristics:
       - starts with ``deepseek-`` → provider="deepseek" if the alias is
         known in the public API catalogue; otherwise ``private`` since
         that is where the user's gateway hosts the full matrix.
       - any other name → ``private`` (matches pro.env L5-17).
    """
    pro_env = _read_pro_env(env_path)
    aliases = _parse_pro_env_aliases(env_path)

    provider = (
        (provider_arg or "").strip().lower()
        or os.environ.get("VIB_LLM_PROVIDER", "").strip().lower()
        or ""
    )

    # --- Model id alias expansion ------------------------------------------------
    model_id = logical_model.strip()
    # 1. Literal match in pro.env TSV block takes highest precedence.
    if model_id in aliases:
        model_id = aliases[model_id]
    # 2. User wrote the bare short name (e.g. "deepseek-v4-flash") and the
    #    TSV block only exposes tagged forms like "(preview)" or
    #    "0817 (正式版)". We prefer the stable release tag (see
    #    _PREFERRED_TAGS) over the preview.
    elif model_id in _PREFERRED_MODEL_REDIRECTS:
        preferred_id = _PREFERRED_MODEL_REDIRECTS[model_id]
        known_ids = set(aliases.values())
        if preferred_id in known_ids:
            # Happy path: the stable release id is listed, use it.
            model_id = preferred_id
        else:
            # Fall back: search the aliases for any short name + a preferred
            # tag, e.g. "deepseek-v4-flash 0817 (正式版)".
            found: Optional[str] = None
            for lhs, rhs in aliases.items():
                # strip any bracketed qualifier
                lhs_strip = re.sub(r"\s*\([^)]*\)\s*$", "", lhs).strip()
                if lhs_strip != model_id:
                    continue
                tag_text = (lhs[len(lhs_strip):]).strip(" ()")
                for tag in _PREFERRED_TAGS:
                    if tag in tag_text:
                        found = rhs
                        break
                if found:
                    break
            model_id = found or model_id
    # 3. Exact legacy compat: deepseek-chat stays as-is (public DeepSeek API)
    elif logical_model == "deepseek-chat":
        pass

    # --- Provider inference ------------------------------------------------------
    if not provider:
        if model_id.startswith("deepseek-") and model_id in ("deepseek-chat", "deepseek-reasoner"):
            provider = "deepseek"
        else:
            # pro.env model catalogue lives on the private gateway.
            provider = "private"

    if provider not in BUILTIN_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Known providers: {sorted(BUILTIN_PROVIDERS)}"
        )

    cfg = BUILTIN_PROVIDERS[provider]

    # --- base URL ---------------------------------------------------------------
    base_url_env = cfg.get("base_url_env")
    base_url = ""
    if base_url_env:
        base_url = os.environ.get(base_url_env, "").strip() or pro_env.get(base_url_env, "")
    if not base_url:
        base_url = cfg["base_url"]

    # --- API key ----------------------------------------------------------------
    key_env = cfg["api_key_env"]
    if provider == "private":
        key_env = (
            os.environ.get("VIB_PRIVATE_API_KEY_ENV", "").strip()
            or pro_env.get("VIB_PRIVATE_API_KEY_ENV", "").strip()
            or key_env
        )
        if not re.fullmatch(r"PRIVATE_API_KEY(?:_\d+)?", key_env):
            raise ValueError("VIB_PRIVATE_API_KEY_ENV must name a PRIVATE_API_KEY variable")
    api_key = os.environ.get(key_env, "").strip() or pro_env.get(key_env, "")
    if not api_key:
        raise ValueError(
            f"Missing API key for provider '{provider}'. Set the {key_env} env var "
            f"or add the key to {env_path}."
        )

    return ResolvedModel(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        label=cfg["label"],
    )


# ---------------------------------------------------------------------------
# Backwards-compat wrappers. Keep old scripts running without edits.
# ---------------------------------------------------------------------------

def resolve_deepseek_api_key(
    api_key: Optional[str] = None,
    fallback_env_path: str = DEFAULT_PRO_ENV,
) -> Optional[str]:
    """Resolve DeepSeek API key from:

    1. explicit argument
    2. DEEPSEEK_API_KEY environment variable
    3. local env-style config file, e.g. configs/pro.env
    """
    if api_key:
        return api_key
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    return _read_pro_env(fallback_env_path).get("DEEPSEEK_API_KEY")


def list_known_provider_aliases(env_path: str = DEFAULT_PRO_ENV) -> Dict[str, List[str]]:
    """Human-readable summary used by ``--help`` output."""
    aliases = _parse_pro_env_aliases(env_path)
    out: Dict[str, List[str]] = {p: [] for p in BUILTIN_PROVIDERS}
    for name in ("deepseek-chat", "deepseek-reasoner"):
        out["deepseek"].append(name)
    for alias in sorted(aliases):
        model_id = aliases[alias]
        out["private"].append(f"{alias} -> {model_id}")
    return out


# ---------------------------------------------------------------------------
# Main client.
# ---------------------------------------------------------------------------

class LLMClient:
    """
    OpenAI-compatible chat completion client backed by any provider.
    Replacement for the previous ``DeepSeekClient`` (still exported as an
    alias, see below).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        env_path: str = DEFAULT_PRO_ENV,
        _resolved: Optional[ResolvedModel] = None,
        ledger_path: Optional[str] = None,
    ):
        """
        Initialise the client.

        If ``_resolved`` is supplied (internal callers) it is used verbatim
        and ``api_key``/``base_url``/``model``/``provider`` are ignored.
        Otherwise the arguments go through the pro.env-driven resolution
        pipeline (see :func:`_resolve_provider`).
        """
        if _resolved is None:
            # Allow a call-site to bypass resolution entirely by passing
            # explicit api_key + base_url (e.g. the legacy DeepSeekClient
            # signature). We still route through _resolve_provider when no
            # explicit override is given.
            if api_key and base_url:
                resolved = ResolvedModel(
                    provider=provider or "custom",
                    model_id=model,
                    base_url=base_url,
                    api_key=api_key,
                    label="Caller-provided endpoint",
                )
            else:
                resolved = _resolve_provider(provider, model, env_path=env_path)
                if api_key:
                    resolved.api_key = api_key
                if base_url:
                    resolved.base_url = base_url
        else:
            resolved = _resolved

        self.resolved = resolved
        self.provider: str = resolved.provider
        self.model: str = resolved.model_id
        self.base_url: str = resolved.base_url
        self.api_key: str = resolved.api_key
        self.inference_protocol = InferenceProtocol(temperature=temperature, max_tokens=max_tokens)
        self.temperature = 0.7 if temperature is None else temperature
        self.max_tokens = 1024 if max_tokens is None else max_tokens

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.request_ledger = RequestLedger(ledger_path)
        self.request_context: Dict[str, str] = {}
        self._stats_offset = 0

        self._client = None

    # ------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=0,
                )
            except ImportError as exc:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Dict:
        """Send a chat completion request.

        Returns a dict with ``content``, ``prompt_tokens``,
        ``completion_tokens``, ``total_tokens``. Transient failures are retried
        with exponential backoff. Legacy numeric counters report known usage
        only; ``usage_status`` and the request ledger distinguish unknown cost.
        Error messages exclude exception text to avoid leaking credentials.
        """
        max_attempts = 3
        backoff_base = 1.5
        last_exc: Optional[Exception] = None
        operation_id = uuid.uuid4().hex

        for attempt in range(max_attempts):
            try:
                return self._chat_once(
                    messages, temperature, max_tokens,
                    operation_id=operation_id, attempt=attempt + 1, stage=stage,
                )
            except LedgerError:
                raise
            except Exception as exc:  # pragma: no cover - network fence
                last_exc = exc
                print(json.dumps({
                    "event": "llm_request_error", "provider": self.provider,
                    "model": self.model, "operation_id": operation_id,
                    "error_type": type(exc).__name__, "attempt": attempt + 1,
                }))
                if getattr(exc, "status_code", None) in (401, 403, 429):
                    break
                if attempt < max_attempts - 1:
                    time.sleep(backoff_base ** attempt)

        return {
            "content": f"[API Error: {type(last_exc).__name__}]",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "usage_status": "unknown",
            "operation_id": operation_id,
        }

    # ------------------------------------------------------------------
    def chat_single_attempt(
        self, messages, temperature, max_tokens, *, operation_id, stage=None,
    ) -> Dict:
        """Send exactly once for an externally reserved, manifest-bound job.

        Unlike ``chat``, failures propagate without automatic retries. The
        request ledger still records every response and unresolved attempt.
        """
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("A frozen operation_id is required")
        return self._chat_once(
            messages, temperature, max_tokens, operation_id=operation_id, stage=stage,
        )

    # ------------------------------------------------------------------
    def _chat_once(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        *,
        operation_id: Optional[str] = None,
        attempt: int = 1,
        stage: Optional[str] = None,
    ) -> Dict:
        parameters = {
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        InferenceProtocol(**parameters)
        request_id = self.request_ledger.start(
            {
                "provider": self.provider, "model": self.model, "base_url": self.base_url,
                "prompt_sha256": fingerprint(messages), "parameters": parameters,
                "context": dict(self.request_context), "stage": stage,
            },
            operation_id=operation_id or uuid.uuid4().hex, attempt=attempt,
        )
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **parameters,
            )
        except Exception as exc:
            self.request_ledger.finish(
                request_id, status="failed", error_type=type(exc).__name__,
                elapsed_seconds=time.monotonic() - started,
            )
            raise

        raw_usage = getattr(response, "usage", None)
        prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
        completion_tokens = getattr(raw_usage, "completion_tokens", None)
        usage_known = all(isinstance(count, int) and not isinstance(count, bool) and count >= 0
                          for count in (prompt_tokens, completion_tokens))
        usage = ({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens} if usage_known else None)
        provider_usage = raw_usage.model_dump(mode="json") if hasattr(raw_usage, "model_dump") else None
        choices = getattr(response, "choices", None)
        content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        valid_content = isinstance(content, str) and bool(content.strip()) and finish_reason != "length"
        status = "completed" if valid_content else "invalid_response"
        if finish_reason == "length":
            status = "truncated"
        self.request_ledger.finish(
            request_id, status=status,
            response_id=getattr(response, "id", None),
            provider_request_id=getattr(response, "_request_id", None),
            served_model=getattr(response, "model", None),
            output_sha256=fingerprint(content),
            usage=usage, usage_status="known" if usage_known else "unknown",
            provider_usage=provider_usage, finish_reason=finish_reason,
            elapsed_seconds=time.monotonic() - started,
        )
        self.get_token_stats()
        if not valid_content:
            raise ValueError("empty or truncated completion")

        return {
            "content": content,
            "prompt_tokens": usage["prompt_tokens"] if usage else 0,
            "completion_tokens": usage["completion_tokens"] if usage else 0,
            "total_tokens": usage["total_tokens"] if usage else 0,
            "usage_status": "known" if usage_known else "unknown",
            "request_id": request_id,
            "operation_id": operation_id,
            "response_id": getattr(response, "id", None),
            "provider_request_id": getattr(response, "_request_id", None),
            "served_model": getattr(response, "model", None),
            "finish_reason": finish_reason,
            "output_sha256": fingerprint(content),
        }

    # ------------------------------------------------------------------
    def get_token_stats(self) -> Dict:
        stats = self.request_ledger.summary(self._stats_offset)
        self.total_calls = stats["total_calls"]
        self.total_prompt_tokens = stats["total_prompt_tokens"]
        self.total_completion_tokens = stats["total_completion_tokens"]
        return stats

    def get_request_records(self) -> List[Dict]:
        """Return all request evidence, including retries and pre-reset attempts."""
        return self.request_ledger.records()

    def reset_stats(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self._stats_offset = len(self.request_ledger.records())


# Alias: old callers ``from llm_client import DeepSeekClient`` keep working
# with the new provider-switch semantics; ``model`` param drives provider
# selection heuristics when no explicit provider given.
DeepSeekClient = LLMClient


def build_client(
    model: str = "deepseek-chat",
    provider: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    env_path: str = DEFAULT_PRO_ENV,
    log: bool = True,
    ledger_path: Optional[str] = None,
) -> LLMClient:
    """Factory used by experimental entry-points.

    Example::

        client = build_client("deepseek-v4-flash")
        # -> automatically picks provider=private, resolves to
        #    deepseek-v4-flash-0817 against PRIVATE_API_URL.

    Environment-driven defaults (stronger than ``model="deepseek-chat"`` sentinel):
      - ``VIB_LLM_MODEL``   — when caller did not override the default model,
        replace the sentinel ``deepseek-chat`` with this value. Use to direct
        all unparameterised callers to the private gateway's production model.
      - ``VIB_LLM_PROVIDER`` — consumed by :func:`_resolve_provider`.
    """
    if model == "deepseek-chat":
        env_model = (os.environ.get("VIB_LLM_MODEL") or "").strip()
        if env_model:
            model = env_model
    resolved = _resolve_provider(provider, model, env_path=env_path)
    if log:
        print(f"[LLM] {resolved.describe()}")
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return LLMClient(_resolved=resolved, ledger_path=ledger_path, **kwargs)
