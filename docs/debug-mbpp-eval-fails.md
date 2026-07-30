[OPEN] Debug Session: mbpp-eval-fails

## Symptom

- `experiments/run_multi_benchmark.py --datasets mbpp` 运行过程中，MBPP 评测几乎全为 ❌（Full/VIB 都失败），且偶发 DeepSeek Connection error。

## Expected

- MBPP 应出现一定比例的 ✅（至少在少量样例上可通过），并且失败应能定位到是“模型输出格式问题/评测器问题/用例拼接问题”等明确原因之一。

## Repro Command

- `python3 experiments/run_multi_benchmark.py --datasets mbpp --limit 5`

## Hypotheses (falsifiable)

1. 模型最终输出不是“纯可执行代码”，包含解释/Markdown/多段文本，导致 `exec()` 直接失败。
2. MBPP 的 `test_list` 不是可直接 `exec()` 的测试脚本（缺少 `check()` 或需要额外上下文），当前 `CodeEvaluator` 拼接方式不匹配。
3. `entry_point` 提取策略对 MBPP 不可靠，导致错误调用或未调用到目标函数，评测恒失败。
4. `DeepSeekClient(max_tokens=256)` 对代码任务过小，输出被截断导致语法错误或逻辑不完整，评测恒失败。
5. 评测进程隔离/超时机制导致真实异常被吞掉（当前只返回 True/False），看起来像“全错”，但根因分布在多种异常上。

## Evidence Plan

- 启动 Debug Server 收集 NDJSON 事件日志。
- 对每个 MBPP 样例记录：
  - entry_point、test_cases 片段、模型输出前 200/后 200 字符、是否包含 ```、是否疑似截断
  - exec 失败的异常类型与 traceback（采样或截断）
  - DeepSeek API 的 request/response metadata（不包含密钥）

## Status

- Next: add instrumentation + reproduce on a few MBPP samples.

