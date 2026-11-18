import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from src.pre_generation_scheduling import (
    count_matched_random_scheduler,
    fixed_round_scheduler,
    parse_history_roles,
    random_skip_scheduler,
    role_coverage_scheduler,
)


def test_parse_history_roles_extracts_roles_and_digest_tags():
    history = [
        "[Planner]: strategy",
        "[Solver-digest]: compute first",
        "[Verifier]: check",
        "unstructured note without a tag",
    ]

    assert parse_history_roles(history) == {"Planner", "Solver", "Verifier"}


def test_role_coverage_scheduler_runs_all_roles_in_first_round():
    schedule = role_coverage_scheduler()
    agents = ["Planner", "Solver", "Verifier"]

    assert schedule("goal", [], 1, agents) == agents


def test_role_coverage_scheduler_only_reruns_unrepresented_roles():
    schedule = role_coverage_scheduler()
    agents = ["Planner", "Solver", "Verifier"]

    chosen = schedule("goal", ["[Planner]: strategy", "[Solver]: compute"], 2, agents)

    assert chosen == ["Verifier"]


def test_role_coverage_scheduler_always_includes_closer_roles():
    schedule = role_coverage_scheduler(closer_roles=["Verifier"])
    agents = ["Planner", "Solver", "Verifier"]
    history = ["[Planner]: a", "[Solver]: b", "[Verifier]: c"]

    assert schedule("goal", history, 2, agents) == ["Verifier"]


def test_random_skip_scheduler_is_deterministic_for_same_task_and_round():
    agents = ["Planner", "Solver", "Verifier"]
    schedule = random_skip_scheduler(keep_rate=0.5, seed=7, task_salt="HumanEval/1")

    first = schedule("goal", [], 2, agents)
    second = schedule("goal", [], 2, agents)

    assert first == second


def test_random_skip_scheduler_varies_across_rounds():
    agents = ["Planner", "Solver", "Verifier"]
    schedule = random_skip_scheduler(keep_rate=0.5, seed=7, task_salt="HumanEval/1")

    patterns = {tuple(schedule("goal", [], step, agents)) for step in range(1, 12)}

    assert len(patterns) > 1


def test_fixed_round_scheduler_runs_all_roles_within_horizon():
    schedule = fixed_round_scheduler(max_rounds=2)
    agents = ["Planner", "Solver", "Verifier"]

    assert schedule("goal", [], 1, agents) == agents
    assert schedule("goal", [], 2, agents) == agents


def test_fixed_round_scheduler_stops_after_horizon():
    schedule = fixed_round_scheduler(max_rounds=2)
    agents = ["Planner", "Solver", "Verifier"]

    assert schedule("goal", [], 3, agents) == []


def test_count_matched_random_scheduler_keeps_base_count_and_is_deterministic():
    base = role_coverage_scheduler()
    schedule = count_matched_random_scheduler(base, seed=7, task_salt="MATH/1")
    agents = ["Planner", "Solver", "Verifier"]
    history = ["[Planner]: plan"]  # coverage keeps 2 of 3

    first = schedule("goal", history, 2, agents)
    second = schedule("goal", history, 2, agents)

    assert first == second
    assert len(first) == 2
    assert set(first).issubset(set(agents))


def test_count_matched_random_scheduler_defers_to_base_when_all_or_none():
    base = role_coverage_scheduler()
    schedule = count_matched_random_scheduler(base, seed=7, task_salt="MATH/1")
    agents = ["Planner", "Solver", "Verifier"]

    # Round 1: coverage keeps everyone -> control also keeps everyone.
    assert schedule("goal", [], 1, agents) == agents
    # Coverage that selects nobody -> control selects nobody.
    empty_base = fixed_round_scheduler(max_rounds=0)
    none_schedule = count_matched_random_scheduler(empty_base, seed=7, task_salt="MATH/1")
    assert none_schedule("goal", [], 1, agents) == []
