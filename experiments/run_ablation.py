#!/usr/bin/env python3
"""
Milestone 4: Ablation Study - Parameter Sensitivity Analysis

This experiment tests different β₀ and λ parameter combinations to find
the Pareto frontier between Token savings and task performance.
"""

import sys
import os
import json
import time
import itertools
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import create_llm_agents
from vib_router import VIBSparseCommRouter
from workflow import VIBMultiAgentWorkflow


# ============================================
# Ablation Parameter Grid
# ============================================

# β₀ controls initial exploration weight
BETA_0_VALUES = [1.0, 2.0, 3.0, 4.0]

# λ controls decay speed (how fast the gate tightens)
LAMBDA_VALUES = [0.05, 0.15, 0.30, 0.50]

# Standard task for ablation (consistent across runs)
ABLATION_TASK = {
    "id": "ablation_sort",
    "name": "排序算法优化（消融实验标准任务）",
    "goal": "设计一个高性能的Python排序函数，要求：1) 对小数组使用插入排序，大数组使用快速排序；2) 处理大量重复元素的情况；3) 给出时间复杂度分析。",
    "max_steps": 4
}


def run_ablation_single(
    api_key: str,
    beta_0: float,
    lambda_decay: float,
    task: dict = None,
    verbose: bool = False
) -> dict:
    """
    Run a single ablation experiment with specific parameters.
    
    Args:
        api_key: DeepSeek API key
        beta_0: β₀ parameter value
        lambda_decay: λ parameter value
        task: Task configuration (uses default if None)
        verbose: Whether to print progress
        
    Returns:
        Result dict with metrics
    """
    if task is None:
        task = ABLATION_TASK
    
    # Initialize LLM client
    llm_client = DeepSeekClient(
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=256
    )
    
    # Create agents
    agents = create_llm_agents(llm_client, roles=["Programmer", "Reviewer", "Architect"])
    
    # Create VIB router with specific parameters
    vib_router = VIBSparseCommRouter(
        beta_0=beta_0,
        lambda_decay=lambda_decay,
        temp_tau=0.2,
        gamma=0.0
    )
    
    # Create and run workflow
    workflow = VIBMultiAgentWorkflow(
        agents=agents,
        vib_router=vib_router,
        max_steps=task["max_steps"]
    )
    
    start_time = time.time()
    result = workflow.run(goal=task["goal"], verbose=verbose)
    elapsed_time = time.time() - start_time
    
    # Get token stats
    token_stats = llm_client.get_token_stats()
    
    # Calculate metrics
    total_candidates = result["stats"]["total_candidates"]
    total_approved = result["stats"]["total_approved"]
    total_blocked = result["stats"]["total_blocked"]
    compression_rate = result["compression_rate"]
    
    # Calculate theoretical token savings
    # In full communication (no VIB), all candidates would be in history
    # With VIB, only approved ones are broadcast
    # Token savings ≈ blocked / total * 100%
    
    return {
        "beta_0": beta_0,
        "lambda_decay": lambda_decay,
        "total_candidates": total_candidates,
        "total_approved": total_approved,
        "total_blocked": total_blocked,
        "compression_rate": compression_rate,
        "total_tokens": token_stats["total_tokens"],
        "prompt_tokens": token_stats["total_prompt_tokens"],
        "completion_tokens": token_stats["total_completion_tokens"],
        "api_calls": token_stats["total_calls"],
        "elapsed_seconds": elapsed_time,
        "history_per_step": result["stats"]["history_per_step"]
    }


def run_full_ablation(
    api_key: str,
    beta_values: list = None,
    lambda_values: list = None,
    verbose: bool = True
) -> list:
    """
    Run full ablation study across parameter grid.
    
    Args:
        api_key: DeepSeek API key
        beta_values: List of β₀ values to test
        lambda_values: List of λ values to test
        verbose: Whether to print progress
        
    Returns:
        List of result dicts
    """
    if beta_values is None:
        beta_values = BETA_0_VALUES
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES
    
    # Generate all parameter combinations
    param_grid = list(itertools.product(beta_values, lambda_values))
    total_runs = len(param_grid)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Milestone 4: Ablation Study")
        print(f"  Parameter Grid: {len(beta_values)} β₀ × {len(lambda_values)} λ = {total_runs} runs")
        print(f"  β₀ values: {beta_values}")
        print(f"  λ values:  {lambda_values}")
        print(f"{'='*70}\n")
    
    results = []
    
    for i, (beta_0, lambda_decay) in enumerate(param_grid, 1):
        if verbose:
            print(f"[{i}/{total_runs}] Running β₀={beta_0}, λ={lambda_decay}...", end=" ")
        
        try:
            result = run_ablation_single(
                api_key=api_key,
                beta_0=beta_0,
                lambda_decay=lambda_decay,
                verbose=False
            )
            results.append(result)
            
            if verbose:
                print(f"Compression: {result['compression_rate']:.1f}% | "
                      f"Tokens: {result['total_tokens']}")
        
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "beta_0": beta_0,
                "lambda_decay": lambda_decay,
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(1)
    
    return results


def analyze_ablation_results(results: list) -> dict:
    """
    Analyze ablation results and find optimal parameters.
    
    Returns:
        Analysis dict with Pareto frontier and recommendations
    """
    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {"error": "No valid results"}
    
    # Sort by compression rate (higher is better for token savings)
    by_compression = sorted(valid_results, key=lambda x: x["compression_rate"], reverse=True)
    
    # Sort by total tokens (lower is better)
    by_tokens = sorted(valid_results, key=lambda x: x["total_tokens"])
    
    # Find Pareto-optimal configurations
    # (high compression, low token usage)
    pareto_front = []
    for r in valid_results:
        dominated = False
        for other in valid_results:
            if (other["compression_rate"] >= r["compression_rate"] and 
                other["total_tokens"] <= r["total_tokens"] and
                (other["compression_rate"] > r["compression_rate"] or 
                 other["total_tokens"] < r["total_tokens"])):
                dominated = True
                break
        if not dominated:
            pareto_front.append(r)
    
    # Calculate averages by parameter
    beta_avg = {}
    for beta in BETA_0_VALUES:
        beta_results = [r for r in valid_results if r["beta_0"] == beta]
        if beta_results:
            beta_avg[beta] = {
                "avg_compression": sum(r["compression_rate"] for r in beta_results) / len(beta_results),
                "avg_tokens": sum(r["total_tokens"] for r in beta_results) / len(beta_results)
            }
    
    lambda_avg = {}
    for lam in LAMBDA_VALUES:
        lam_results = [r for r in valid_results if r["lambda_decay"] == lam]
        if lam_results:
            lambda_avg[lam] = {
                "avg_compression": sum(r["compression_rate"] for r in lam_results) / len(lam_results),
                "avg_tokens": sum(r["total_tokens"] for r in lam_results) / len(lam_results)
            }
    
    return {
        "total_valid_runs": len(valid_results),
        "best_compression": by_compression[0] if by_compression else None,
        "lowest_tokens": by_tokens[0] if by_tokens else None,
        "pareto_front": pareto_front,
        "beta_averages": beta_avg,
        "lambda_averages": lambda_avg
    }


def print_ablation_report(results: list, analysis: dict):
    """Print formatted ablation study report."""
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY REPORT")
    print(f"{'='*70}")
    
    print(f"\n  Total Valid Runs: {analysis['total_valid_runs']}")
    
    if analysis.get("best_compression"):
        best = analysis["best_compression"]
        print(f"\n  🏆 Best Compression Rate:")
        print(f"     β₀={best['beta_0']}, λ={best['lambda_decay']}")
        print(f"     Compression: {best['compression_rate']:.1f}%")
        print(f"     Tokens: {best['total_tokens']}")
    
    if analysis.get("lowest_tokens"):
        lowest = analysis["lowest_tokens"]
        print(f"\n  💰 Lowest Token Usage:")
        print(f"     β₀={lowest['beta_0']}, λ={lowest['lambda_decay']}")
        print(f"     Tokens: {lowest['total_tokens']}")
        print(f"     Compression: {lowest['compression_rate']:.1f}%")
    
    print(f"\n  --- Parameter Sensitivity: β₀ ---")
    print(f"  {'β₀':<6} {'Avg Compression':<18} {'Avg Tokens':<12}")
    print(f"  {'-'*36}")
    for beta, stats in sorted(analysis.get("beta_averages", {}).items()):
        print(f"  {beta:<6.1f} {stats['avg_compression']:<18.1f}% {stats['avg_tokens']:<12.0f}")
    
    print(f"\n  --- Parameter Sensitivity: λ ---")
    print(f"  {'λ':<6} {'Avg Compression':<18} {'Avg Tokens':<12}")
    print(f"  {'-'*36}")
    for lam, stats in sorted(analysis.get("lambda_averages", {}).items()):
        print(f"  {lam:<6.2f} {stats['avg_compression']:<18.1f}% {stats['avg_tokens']:<12.0f}")
    
    if analysis.get("pareto_front"):
        print(f"\n  --- Pareto Frontier (Optimal Configurations) ---")
        for r in analysis["pareto_front"]:
            print(f"     β₀={r['beta_0']}, λ={r['lambda_decay']}: "
                  f"Compression={r['compression_rate']:.1f}%, Tokens={r['total_tokens']}")
    
    print(f"\n{'='*70}\n")


def main():
    """Run Milestone 4 ablation study."""
    # Get API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        sys.exit(1)
    
    # For quick testing, use smaller grid
    # Full grid: BETA_0_VALUES × LAMBDA_VALUES = 16 runs
    # Quick grid: 3 × 3 = 9 runs
    quick_mode = "--quick" in sys.argv
    
    if quick_mode:
        beta_values = [1.5, 2.0, 3.0]
        lambda_values = [0.10, 0.15, 0.30]
        print("\n[Quick Mode: Reduced parameter grid]")
    else:
        beta_values = BETA_0_VALUES
        lambda_values = LAMBDA_VALUES
    
    # Run ablation
    results = run_full_ablation(
        api_key=api_key,
        beta_values=beta_values,
        lambda_values=lambda_values,
        verbose=True
    )
    
    # Analyze results
    analysis = analyze_ablation_results(results)
    
    # Print report
    print_ablation_report(results, analysis)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "analysis": analysis,
            "params": {
                "beta_values": beta_values,
                "lambda_values": lambda_values,
                "task": ABLATION_TASK
            }
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  Results saved to: {output_file}")
    
    return results, analysis


if __name__ == "__main__":
    main()
