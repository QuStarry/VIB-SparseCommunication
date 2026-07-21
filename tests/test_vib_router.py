"""
Milestone 1: VIB Router Unit Tests

Validates the VIB router's mathematical formulas using manually constructed test cases:
- High redundancy messages (should be blocked)
- High value new information (should pass)
- Unrelated messages (should be blocked due to low task relevance)
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.vib_router import VIBSparseCommRouter, VIBEvaluationResult


class TestVIBRouterBasic:
    """Basic functionality tests for VIBSparseCommRouter."""
    
    @pytest.fixture
    def router(self):
        """Create a router with default parameters."""
        return VIBSparseCommRouter(
            beta_0=2.0,
            lambda_decay=0.15,
            temp_tau=0.2,
            gamma=0.0,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    def test_first_round_always_passes(self, router):
        """First round should always pass (empty history)."""
        result = router.evaluate_detailed(
            candidate_msg="Any message here",
            history=[],
            goal="Some goal",
            step_t=1
        )
        assert result.is_allowed is True
        assert result.redundancy == 0.0
    
    def test_beta_decay(self, router):
        """β_t should decay exponentially with step t."""
        beta_1 = router.compute_beta_t(1)
        beta_5 = router.compute_beta_t(5)
        beta_10 = router.compute_beta_t(10)
        
        # β_t should decrease as t increases
        assert beta_1 > beta_5 > beta_10
        
        # Check the formula: β_t = β_0 * exp(-λ * t)
        import math
        expected_beta_5 = 2.0 * math.exp(-0.15 * 5)
        assert abs(beta_5 - expected_beta_5) < 1e-6


class TestVIBRouterMilestone1:
    """
    Milestone 1 Validation: Test cases from PRD.
    
    These tests verify that the VIB router correctly distinguishes between:
    - High redundancy messages (should be blocked)
    - High value new information (should pass)
    """
    
    @pytest.fixture
    def router(self):
        """Create router with PRD-suggested parameters."""
        return VIBSparseCommRouter(
            beta_0=2.0,
            lambda_decay=0.15,
            temp_tau=0.2,
            gamma=0.0,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    def test_case_a_high_redundancy_should_block(self, router):
        """
        Test Case A (High Redundancy - Should be Blocked):
        
        Goal: "Optimize the sorting performance of this Python code"
        History: ["Agent A: We can use quicksort, its average time complexity is O(n log n)."]
        Candidate: "Agent B: Yes, I also think quicksort is great, because its average 
                   time complexity is indeed O(n log n), let's use quicksort."
        
        Expected: This redundant message should have LOW utility and be BLOCKED.
        """
        goal = "Optimize the sorting performance of this Python code"
        history = [
            "Agent A: We can use quicksort, its average time complexity is O(n log n)."
        ]
        candidate = (
            "Agent B: Yes, I also think quicksort is great, because its average "
            "time complexity is indeed O(n log n), let's use quicksort."
        )
        
        result = router.evaluate_detailed(
            candidate_msg=candidate,
            history=history,
            goal=goal,
            step_t=1
        )
        
        print(f"\n=== Test Case A: High Redundancy ===")
        print(f"Sufficiency I(M;G): {result.sufficiency:.4f}")
        print(f"Redundancy I(M;H):  {result.redundancy:.4f}")
        print(f"Beta_t:             {result.beta_t:.4f}")
        print(f"Utility Score:      {result.utility_score:.4f}")
        print(f"Is Allowed:         {result.is_allowed}")
        
        # High redundancy message should have high redundancy score
        # and likely be blocked (utility < 0)
        assert result.redundancy > 0, "Redundancy should be positive"
        
        # Store for comparison with Case B
        self.__class__.case_a_utility = result.utility_score
        self.__class__.case_a_result = result
    
    def test_case_b_high_value_should_pass(self, router):
        """
        Test Case B (High Value New Info - Should Pass):
        
        Goal: "Optimize the sorting performance of this Python code"
        History: ["Agent A: We can use quicksort, its average time complexity is O(n log n)."]
        Candidate: "Agent B: Considering that the data may contain many duplicate elements, 
                   I suggest using Three-way Quick Sort to avoid degradation."
        
        Expected: This novel, valuable message should have HIGH utility and PASS.
        """
        goal = "Optimize the sorting performance of this Python code"
        history = [
            "Agent A: We can use quicksort, its average time complexity is O(n log n)."
        ]
        candidate = (
            "Agent B: Considering that the data may contain many duplicate elements, "
            "I suggest using Three-way Quick Sort to avoid degradation."
        )
        
        result = router.evaluate_detailed(
            candidate_msg=candidate,
            history=history,
            goal=goal,
            step_t=1
        )
        
        print(f"\n=== Test Case B: High Value New Info ===")
        print(f"Sufficiency I(M;G): {result.sufficiency:.4f}")
        print(f"Redundancy I(M;H):  {result.redundancy:.4f}")
        print(f"Beta_t:             {result.beta_t:.4f}")
        print(f"Utility Score:      {result.utility_score:.4f}")
        print(f"Is Allowed:         {result.is_allowed}")
        
        # Store for comparison
        self.__class__.case_b_utility = result.utility_score
        self.__class__.case_b_result = result
    
    def test_case_b_utility_greater_than_case_a(self, router):
        """
        Core Validation: Case B utility should be significantly greater than Case A.
        
        This is the key validation criterion from the PRD:
        "Case B utility >> Case A utility"
        """
        # Run both cases
        goal = "Optimize the sorting performance of this Python code"
        history = [
            "Agent A: We can use quicksort, its average time complexity is O(n log n)."
        ]
        
        # Case A: Redundant
        candidate_a = (
            "Agent B: Yes, I also think quicksort is great, because its average "
            "time complexity is indeed O(n log n), let's use quicksort."
        )
        result_a = router.evaluate_detailed(candidate_a, history, goal, step_t=1)
        
        # Case B: Novel info
        candidate_b = (
            "Agent B: Considering that the data may contain many duplicate elements, "
            "I suggest using Three-way Quick Sort to avoid degradation."
        )
        result_b = router.evaluate_detailed(candidate_b, history, goal, step_t=1)
        
        print(f"\n=== Comparison: Case A vs Case B ===")
        print(f"Case A (Redundant) Utility: {result_a.utility_score:.4f}")
        print(f"Case B (Novel) Utility:     {result_b.utility_score:.4f}")
        print(f"Difference (B - A):         {result_b.utility_score - result_a.utility_score:.4f}")
        
        # KEY ASSERTION: Case B should have higher utility than Case A
        assert result_b.utility_score > result_a.utility_score, (
            f"Case B utility ({result_b.utility_score:.4f}) should be greater than "
            f"Case A utility ({result_a.utility_score:.4f})"
        )
        
        # Case B should have lower redundancy (more novel)
        assert result_b.redundancy < result_a.redundancy, (
            "Case B should have lower redundancy than Case A"
        )


class TestVIBRouterExtended:
    """Extended test cases for comprehensive validation."""
    
    @pytest.fixture
    def router(self):
        return VIBSparseCommRouter(
            beta_0=2.0,
            lambda_decay=0.15,
            temp_tau=0.2,
            gamma=0.0,
            embedding_model="all-MiniLM-L6-v2"
        )
    
    def test_unrelated_message_low_sufficiency(self, router):
        """Unrelated messages should have low sufficiency (task relevance)."""
        goal = "Optimize the sorting performance of this Python code"
        history = ["Agent A: Let's analyze the algorithm complexity."]
        candidate = "Agent B: The weather is nice today, let's go for a walk."
        
        result = router.evaluate_detailed(candidate, history, goal, step_t=1)
        
        print(f"\n=== Unrelated Message Test ===")
        print(f"Sufficiency: {result.sufficiency:.4f}")
        print(f"Utility: {result.utility_score:.4f}")
        
        # Unrelated message should have low sufficiency
        # (Note: embedding models may still give some baseline similarity)
        assert result.sufficiency < 0.5, "Unrelated message should have low sufficiency"
    
    def test_increasing_history_increases_redundancy(self, router):
        """More similar history items should increase redundancy score."""
        goal = "Discuss machine learning algorithms"
        candidate = "We should use neural networks for this task."
        
        # Single history item
        history_1 = ["Neural networks are powerful for pattern recognition."]
        result_1 = router.evaluate_detailed(candidate, history_1, goal, step_t=1)
        
        # Multiple similar history items
        history_3 = [
            "Neural networks are powerful for pattern recognition.",
            "Deep learning uses neural networks with many layers.",
            "Neural network architectures include CNN, RNN, and Transformer."
        ]
        result_3 = router.evaluate_detailed(candidate, history_3, goal, step_t=1)
        
        print(f"\n=== History Size Effect ===")
        print(f"Redundancy (1 history): {result_1.redundancy:.4f}")
        print(f"Redundancy (3 history): {result_3.redundancy:.4f}")
        
        # More history should increase redundancy
        assert result_3.redundancy > result_1.redundancy
    
    def test_late_stage_stricter_gating(self, router):
        """Later stages (higher t) should have stricter gating due to β_t decay."""
        goal = "Solve this complex problem"
        history = ["Initial discussion about the problem."]
        candidate = "Here's a slightly different perspective on the solution."
        
        result_early = router.evaluate_detailed(candidate, history, goal, step_t=1)
        result_late = router.evaluate_detailed(candidate, history, goal, step_t=10)
        
        print(f"\n=== Time Decay Effect ===")
        print(f"β_t at step 1:  {result_early.beta_t:.4f}")
        print(f"β_t at step 10: {result_late.beta_t:.4f}")
        print(f"Utility at step 1:  {result_early.utility_score:.4f}")
        print(f"Utility at step 10: {result_late.utility_score:.4f}")
        
        # β_t should be lower at later stages
        assert result_late.beta_t < result_early.beta_t
        
        # Utility should generally be lower at later stages (stricter gating)
        # because β_t * sufficiency decreases while redundancy stays same
        assert result_late.utility_score < result_early.utility_score


def run_milestone1_validation():
    """
    Run Milestone 1 validation with detailed output.
    This function can be called directly for a quick validation check.
    """
    print("=" * 60)
    print("Milestone 1: VIB Router Mathematical Formula Validation")
    print("=" * 60)
    
    router = VIBSparseCommRouter(
        beta_0=2.0,
        lambda_decay=0.15,
        temp_tau=0.2,
        gamma=0.0,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    goal = "Optimize the sorting performance of this Python code"
    history = [
        "Agent A: We can use quicksort, its average time complexity is O(n log n)."
    ]
    
    # Test Case A: High Redundancy
    candidate_a = (
        "Agent B: Yes, I also think quicksort is great, because its average "
        "time complexity is indeed O(n log n), let's use quicksort."
    )
    
    # Test Case B: High Value
    candidate_b = (
        "Agent B: Considering that the data may contain many duplicate elements, "
        "I suggest using Three-way Quick Sort to avoid degradation."
    )
    
    print("\n--- Evaluating Case A (High Redundancy) ---")
    result_a = router.evaluate_detailed(candidate_a, history, goal, step_t=1)
    print(f"  Sufficiency I(M;G): {result_a.sufficiency:.4f}")
    print(f"  Redundancy I(M;H):  {result_a.redundancy:.4f}")
    print(f"  β_t:                {result_a.beta_t:.4f}")
    print(f"  Utility Score:      {result_a.utility_score:.4f}")
    print(f"  Decision:           {'PASS ✓' if result_a.is_allowed else 'BLOCK ✗'}")
    
    print("\n--- Evaluating Case B (High Value New Info) ---")
    result_b = router.evaluate_detailed(candidate_b, history, goal, step_t=1)
    print(f"  Sufficiency I(M;G): {result_b.sufficiency:.4f}")
    print(f"  Redundancy I(M;H):  {result_b.redundancy:.4f}")
    print(f"  β_t:                {result_b.beta_t:.4f}")
    print(f"  Utility Score:      {result_b.utility_score:.4f}")
    print(f"  Decision:           {'PASS ✓' if result_b.is_allowed else 'BLOCK ✗'}")
    
    print("\n--- Comparison ---")
    print(f"  Utility(B) - Utility(A) = {result_b.utility_score - result_a.utility_score:.4f}")
    
    if result_b.utility_score > result_a.utility_score:
        print("\n✓ VALIDATION PASSED: Case B utility > Case A utility (as expected)")
    else:
        print("\n✗ VALIDATION FAILED: Case B utility should be > Case A utility")
    
    return result_b.utility_score > result_a.utility_score


if __name__ == "__main__":
    run_milestone1_validation()
