"""Explicit inference settings with reproducible historical stage defaults."""

from dataclasses import asdict, dataclass
import math
from typing import Optional


STAGE_DEFAULTS = {
    "agent": {"temperature": 0.2, "max_tokens": 256},
    "code": {"temperature": 0.1, "max_tokens": 1024},
    "math": {"temperature": 0.1, "max_tokens": 256},
}


@dataclass(frozen=True)
class InferenceProtocol:
    """Call overrides > stage overrides > global overrides > legacy defaults."""

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    agent_temperature: Optional[float] = None
    agent_max_tokens: Optional[int] = None
    code_temperature: Optional[float] = None
    code_max_tokens: Optional[int] = None
    math_temperature: Optional[float] = None
    math_max_tokens: Optional[int] = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None:
                continue
            if name.endswith("max_tokens"):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            elif not math.isfinite(value) or not 0 <= value <= 2:
                raise ValueError(f"{name} must be finite and between 0 and 2")

    def resolve(self, stage: str) -> dict:
        defaults = STAGE_DEFAULTS[stage]
        result = {}
        for name, default in defaults.items():
            value = getattr(self, f"{stage}_{name}")
            if value is None:
                value = getattr(self, name)
            result[name] = default if value is None else value
        return result

    def describe(self) -> dict:
        """Serializable declarations and effective stage settings."""
        return {
            "version": "explicit_stages_v1",
            "overrides": asdict(self),
            "effective": {stage: self.resolve(stage) for stage in STAGE_DEFAULTS},
        }


def stage_parameters(client, stage: str, *, temperature=None, max_tokens=None) -> dict:
    """Resolve stage parameters for real clients and legacy recording clients."""
    protocol = getattr(client, "inference_protocol", None)
    if protocol is None:
        protocol = InferenceProtocol(
            temperature=getattr(client, "temperature", None),
            max_tokens=getattr(client, "max_tokens", None),
        )
    result = protocol.resolve(stage)
    if temperature is not None:
        result["temperature"] = temperature
    if max_tokens is not None:
        result["max_tokens"] = max_tokens
    InferenceProtocol(**result)
    return result
