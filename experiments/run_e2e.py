#!/usr/bin/env python3
"""
Milestone 3: End-to-End Multi-Agent Collaboration with DeepSeek LLM

This experiment integrates real LLM agents with the VIB sparse communication
mechanism and runs actual collaborative tasks.
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client import DeepSeekClient
from llm_agents import create_llm_agents
from vib_router import VIBSparseCommRouter
from workflow import VIBMultiAgentWorkflow


# ============================================
# Experiment Tasks
# ============================================

EXPERIMENT_TASKS = [
    {
        "id": "sort_optimization",
        "name": "排序算法优化",
        "goal": "设计一个高性能的Python排序函数，要求：1) 对小数组使用插入排序，大数组使用快速排序；2) 处理大量重复元素的情况；3) 给出时间复杂度分析。",
        "max_steps": 4
    },
    {
        "id": "api_design",
        "name": "RESTful API 设计",
        "goal": "为一个在线书店设计RESTful API，要求：1) 支持书籍CRUD操作；2) 支持用户购物车；3) 考虑分页、认证和错误处理。请给出核心端点设计。",
        "max_steps": 4
    },
    {
        "id": "debug_task",
        "name": "代码调试",
        "goal": "以下Python代码存在内存泄漏问题，请分析原因并给出修复方案：\ndef process_data(items):\n    cache = {}\n    results = []\n    for item in items:\n        key = str(item)\n        if key not in cache:\n            cache[key] = expensive_compute(item)\n        results.append(cache[key])\n    return results",
        "max_steps": 4
    }
]


def run_single_experiment(
    task: dict,
    api_key: str,
    beta_0: float = 2.0,
    lambda_decay: float = 0.15,
    verbose: bool = True
) -> dict:
    """
    Run a single end-to-end experiment.
    
    Args:
        task: Task configuration dict
        api_key: DeepSeek API key
        beta_0: VIB beta_0 parameter
        lambda_decay: VIB lambda decay parameter
        verbose: Whether to print progress
        
    Returns:
        Experiment results dict
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Task: {task['name']}")
        print(f"  Goal: {task['goal'][:80]}...")
        print(f"  VIB Params: β₀={beta_0}, λ={lambda_decay}")
        print(f"{'='*70}")
    
    # Initialize LLM client
    llm_client = DeepSeekClient(
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=256
    )
    
    # Create LLM agents
    agents = create_llm_agents(llm_client, roles=["Programmer", "Reviewer", "Architect"])
    
    # Create VIB router
    vib_router = VIBSparseCommRouter(
        beta_0=beta_0,
        lambda_decay=lambda_decay,
        temp_tau=0.2,
        gamma=0.0
    )
    
    # Create workflow
    workflow = VIBMultiAgentWorkflow(
        agents=agents,
        vib_router=vib_router,
        max_steps=task["max_steps"]
    )
    
    # Run experiment
    start_time = time.time()
    result = workflow.run(goal=task["goal"], verbose=verbose)
    elapsed_time = time.time() - start_time
    
    # Get token stats
    token_stats = llm_client.get_token_stats()
    
    # Compile results
    experiment_result = {
        "task_id": task["id"],
        "task_name": task["name"],
        "goal": task["goal"],
        "vib_params": {
            "beta_0": beta_0,
            "lambda_decay": lambda_decay,
            "tau": 0.2,
            "gamma": 0.0
        },
        "stats": result["stats"],
        "compression_rate": result["compression_rate"],
        "token_usage": token_stats,
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    if verbose:
        print(f"\n--- Token Usage ---")
        print(f"  Total API Calls: {token_stats['total_calls']}")
        print(f"  Prompt Tokens:   {token_stats['total_prompt_tokens']}")
        print(f"  Output Tokens:   {token_stats['total_completion_tokens']}")
        print(f"  Total Tokens:    {token_stats['total_tokens']}")
        print(f"  Elapsed Time:    {elapsed_time:.1f}s")
    
    return experiment_result


def run_baseline_experiment(
    task: dict,
    api_key: str,
    verbose: bool = True
) -> dict:
    """
    Run baseline experiment (single agent, no VIB filtering).
    Used for comparison.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  BASELINE: {task['name']}")
        print(f"{'='*70}")
    
    llm_client = DeepSeekClient(
        api_key=api_key,
        model="deepseek-chat",
        temperature=0.7,
        max_tokens=2048
    )
    
    # Single agent, multiple rounds without VIB
    messages = [
        {"role": "system", "content": "你是一位全能型AI助手。请针对给定任务提供完整的解决方案。"},
        {"role": "user", "content": f"任务: {task['goal']}\n\n请给出详细的解决方案。"}
    ]
    
    start_time = time.time()
    result = llm_client.chat(messages=messages, max_tokens=1024)
    elapsed_time = time.time() - start_time
    
    token_stats = llm_client.get_token_stats()
    
    if verbose:
        print(f"\n--- Baseline Response (truncated) ---")
        print(f"  {result['content'][:200]}...")
        print(f"\n--- Token Usage ---")
        print(f"  Total Tokens: {token_stats['total_tokens']}")
        print(f"  Elapsed Time: {elapsed_time:.1f}s")
    
    return {
        "task_id": task["id"],
        "task_name": task["name"],
        "mode": "baseline",
        "response": result["content"],
        "token_usage": token_stats,
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }


def main():
    """Run Milestone 3 end-to-end experiments."""
    # Get API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it with: export DEEPSEEK_API_KEY=your_key_here")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("  Milestone 3: End-to-End Multi-Agent Collaboration Experiment")
    print("  Model: DeepSeek-Chat")
    print("="*70)
    
    all_results = []
    
    # Run experiments for each task
    for task in EXPERIMENT_TASKS:
        # Run VIB sparse communication experiment
        vib_result = run_single_experiment(
            task=task,
            api_key=api_key,
            beta_0=2.0,
            lambda_decay=0.15,
            verbose=True
        )
        all_results.append(vib_result)
        
        # Small delay between API calls
        time.sleep(1)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"milestone3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  Experiment Complete!")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Print summary
    print("\n--- Summary ---")
    for r in all_results:
        print(f"  [{r['task_name']}] Compression: {r['compression_rate']:.1f}% | "
              f"Tokens: {r['token_usage']['total_tokens']} | "
              f"Time: {r['elapsed_seconds']:.1f}s")
    
    return all_results


if __name__ == "__main__":
    main()
