"""Pre-generation scheduling policies for multi-agent workflows.

A pre-generation scheduler decides *which roles may run* before any candidate
message exists, so a skipped role costs zero tokens. This is the structural
difference from post-generation gating, where the candidate has already been
paid for and can only be discarded afterwards.

Every policy here is a pure function of evidence visible at decision time: the
task goal, the approved shared history so far, the round index and the team's
role names. None of them may consult reference answers or any hidden signal.

Scheduler signature::

    scheduler(goal: str, history: List[str], step_t: int,
              agent_names: List[str]) -> List[str]

It returns the subset of ``agent_names`` allowed to generate this round. An
empty list skips the round entirely (no model calls).
"""

import hashlib
import random
import re
from typing import Callable, Iterable, List, Optional, Set

# Approved messages are formatted "[Role]: content"; digests use
# "[Role-digest]: content". Both must map back to the originating role.
_ROLE_RE = re.compile(r"^\[([^\]]+?)(?:-digest)?\]:")


def parse_history_roles(history: Iterable[str]) -> Set[str]:
    """Return the set of roles already represented in the approved history."""
    roles: Set[str] = set()
    for entry in history:
        match = _ROLE_RE.match(entry.strip())
        if match:
            roles.add(match.group(1).strip())
    return roles


def _round_rng(seed: int, task_salt: str, step_t: int) -> random.Random:
    """Deterministic per-(task, round) RNG so replay is reproducible."""
    key = f"{seed}:{task_salt}:{step_t}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def role_coverage_scheduler(
    closer_roles: Optional[Iterable[str]] = None,
    always_first_round: bool = True
) -> Callable:
    """Run only roles not yet represented in the approved history.

    Round 1 runs every role so the team starts from a non-empty context. Later
    rounds run only roles whose messages never reached the shared history, plus
    a mandatory ``closer_roles`` set (e.g. the role that must produce the final
    artefact). Evidence used: history role tags and the round index.
    """
    closers = set(closer_roles or [])

    def schedule(goal: str, history: List[str], step_t: int, agent_names: List[str]) -> List[str]:
        if always_first_round and step_t <= 1:
            return list(agent_names)
        represented = parse_history_roles(history)
        selected = {name for name in agent_names if name not in represented}
        selected |= {name for name in agent_names if name in closers}
        return [name for name in agent_names if name in selected]

    return schedule


def random_skip_scheduler(keep_rate: float, seed: int, task_salt: str = "") -> Callable:
    """Independent Bernoulli keep/skip per role, seeded per task and round.

    Serves as the uniform-cost control for a scheduler: it skips roles without
    using any task evidence, so any accuracy change isolates the scheduling
    signal rather than the mere reduction of calls.
    """
    def schedule(goal: str, history: List[str], step_t: int, agent_names: List[str]) -> List[str]:
        rng = _round_rng(seed, task_salt, step_t)
        return [name for name in agent_names if rng.random() < keep_rate]

    return schedule


def count_matched_random_scheduler(
    base_scheduler: Callable, seed: int, task_salt: str = ""
) -> Callable:
    """Count-matched random control for ``base_scheduler``.

    The base scheduler decides *how many* roles may run this round from visible
    evidence; this control keeps that count but chooses *which* roles run
    uniformly at random. The two arms therefore trim the same number of calls,
    so an accuracy gap isolates the scheduling signal (which roles) from the
    mere reduction in calls (how many). Round 1 keeps every role whenever the
    base already does.
    """
    def schedule(goal: str, history: List[str], step_t: int, agent_names: List[str]) -> List[str]:
        keep = len(base_scheduler(goal, history, step_t, agent_names) or [])
        if keep >= len(agent_names):
            return list(agent_names)
        if keep <= 0:
            return []
        rng = _round_rng(seed, task_salt, step_t)
        kept = set(rng.sample(list(agent_names), keep))
        return [name for name in agent_names if name in kept]

    return schedule


def fixed_round_scheduler(max_rounds: int) -> Callable:
    """Run every role for the first ``max_rounds`` rounds, then stop.

    A trivial trimming baseline: after the horizon it returns no roles, so all
    remaining rounds cost zero tokens.
    """
    def schedule(goal: str, history: List[str], step_t: int, agent_names: List[str]) -> List[str]:
        return list(agent_names) if step_t <= max_rounds else []

    return schedule
