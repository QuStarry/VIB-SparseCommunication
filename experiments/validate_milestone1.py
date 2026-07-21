#!/usr/bin/env python3
"""
Milestone 1 Validation Script
Run this to validate the VIB Router mathematical formulas.
"""

import sys
sys.path.insert(0, 'src')

from vib_router import VIBSparseCommRouter

def main():
    print("=" * 60)
    print("Milestone 1: VIB Router Mathematical Formula Validation")
    print("=" * 60)
    print("\nInitializing router (loading embedding model)...")
    
    router = VIBSparseCommRouter(
        beta_0=2.0,
        lambda_decay=0.15,
        temp_tau=0.2,
        gamma=0.0,
        embedding_model="all-MiniLM-L6-v2"
    )
    print("Router initialized!\n")

    goal = "Optimize the sorting performance of this Python code"
    history = ["Agent A: We can use quicksort, its average time complexity is O(n log n)."]

    # Test Case A: High Redundancy
    candidate_a = "Agent B: Yes, I also think quicksort is great, because its average time complexity is indeed O(n log n), lets use quicksort."

    # Test Case B: High Value
    candidate_b = "Agent B: Considering that the data may contain many duplicate elements, I suggest using Three-way Quick Sort to avoid degradation."

    print("\n--- Evaluating Case A (High Redundancy) ---")
    result_a = router.evaluate_detailed(candidate_a, history, goal, step_t=1)
    print(f"  Sufficiency I(M;G): {result_a.sufficiency:.4f}")
    print(f"  Redundancy I(M;H):  {result_a.redundancy:.4f}")
    print(f"  beta_t:             {result_a.beta_t:.4f}")
    print(f"  Utility Score:      {result_a.utility_score:.4f}")
    print(f"  Decision:           {'PASS' if result_a.is_allowed else 'BLOCK'}")

    print("\n--- Evaluating Case B (High Value New Info) ---")
    result_b = router.evaluate_detailed(candidate_b, history, goal, step_t=1)
    print(f"  Sufficiency I(M;G): {result_b.sufficiency:.4f}")
    print(f"  Redundancy I(M;H):  {result_b.redundancy:.4f}")
    print(f"  beta_t:             {result_b.beta_t:.4f}")
    print(f"  Utility Score:      {result_b.utility_score:.4f}")
    print(f"  Decision:           {'PASS' if result_b.is_allowed else 'BLOCK'}")

    print("\n--- Comparison ---")
    print(f"  Utility(B) - Utility(A) = {result_b.utility_score - result_a.utility_score:.4f}")

    if result_b.utility_score > result_a.utility_score:
        print("\n✓ VALIDATION PASSED: Case B utility > Case A utility (as expected)")
        return True
    else:
        print("\n✗ VALIDATION FAILED: Case B utility should be > Case A utility")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
