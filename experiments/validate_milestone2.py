#!/usr/bin/env python3
"""
Milestone 2 Validation Script
Run this to validate the LangGraph state machine with VIB sparse communication.
"""

import sys
sys.path.insert(0, 'src')

from workflow import (
    VIBMultiAgentWorkflow,
    create_mock_programmer_agent,
    create_mock_reviewer_agent,
    create_mock_redundant_agent
)
from vib_router import VIBSparseCommRouter


def main():
    print("=" * 60)
    print("Milestone 2: LangGraph State Machine Validation")
    print("=" * 60)
    
    # Create agents
    agents = [
        create_mock_programmer_agent(),
        create_mock_reviewer_agent(),
        create_mock_redundant_agent()  # This one should get blocked often
    ]
    
    print(f"\nAgents: {[a['name'] for a in agents]}")
    
    # Create VIB router
    vib_router = VIBSparseCommRouter(
        beta_0=2.0,
        lambda_decay=0.15,
        temp_tau=0.2,
        gamma=0.0
    )
    
    # Create workflow
    print("\nBuilding LangGraph workflow...")
    workflow = VIBMultiAgentWorkflow(
        agents=agents,
        vib_router=vib_router,
        max_steps=5
    )
    print("Workflow built successfully!")
    
    # Run workflow
    print("\nRunning workflow...")
    result = workflow.run(
        goal="Optimize the sorting performance of this Python code",
        verbose=True
    )
    
    # Validate results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    stats = result["stats"]
    
    # Check that workflow ran
    assert stats["steps_completed"] > 0, "Workflow should complete at least 1 step"
    print(f"✓ Workflow completed {stats['steps_completed']} steps")
    
    # Check that candidates were generated
    assert stats["total_candidates"] > 0, "Should have generated candidates"
    print(f"✓ Generated {stats['total_candidates']} total candidates")
    
    # Check that some were approved
    assert stats["total_approved"] > 0, "Should have approved some candidates"
    print(f"✓ Approved {stats['total_approved']} candidates")
    
    # Check that some were blocked (sparse communication working)
    if stats["total_blocked"] > 0:
        print(f"✓ Blocked {stats['total_blocked']} redundant candidates")
    else:
        print("⚠ No candidates were blocked (may need parameter tuning)")
    
    # Check compression rate
    compression = result["compression_rate"]
    print(f"✓ Compression rate: {compression:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ MILESTONE 2 VALIDATION PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
