---
kind: logging_system
name: 日志系统：基于 print 的轻量级调试输出（无结构化日志框架）
category: logging_system
scope:
    - '**'
source_files:
    - run_summary.py
    - src/workflow.py
---

本仓库未引入任何 Python 标准库 logging、loguru、structlog 等日志框架，也未定义 logger 实例或日志级别。所有“日志”行为均通过 `print()` 直接输出到标准输出，属于最轻量的调试/演示输出方式。

证据与分布：
- run_summary.py：大量使用 `print(f"...")` 打印测试用例输入、VIB 指标（sufficiency、redundancy、beta_t、utility_score）、决策结果以及关键验证断言。
- src/workflow.py：在 `run()` 中通过 `verbose=True` 分支打印工作流启动信息、每步 gatekeeper 的 approved/blocked 计数、最终压缩率统计；`run_milestone2_demo()` 也使用 `print` 输出里程碑标题。
- tests/test_vib_router.py：仅包含单元测试逻辑，未见 assert 之外的日志输出。
- src/vib_router.py、src/state.py：核心算法模块无任何 print 或日志调用，保持纯函数式实现。

架构与约定：
- 无集中式日志初始化文件（如 log_config.py），不存在全局 logger 对象。
- 无日志级别（DEBUG/INFO/WARNING/ERROR）区分，所有输出均为同一格式。
- 无结构化字段（JSON/键值对），输出为人类可读的格式化字符串。
- 无日志 sink 配置（文件、stdout/stderr 分离、异步写入等）。
- 仅在顶层脚本和演示入口使用 print，核心库模块不耦合任何 I/O 输出。

开发者应遵循的规则：
1. 若需新增运行时诊断输出，请在顶层脚本或演示函数中使用 `print`，避免在 `src/vib_router.py`、`src/state.py` 等核心模块中引入 I/O。
2. 当前仓库定位是算法原型与演示，不建议在此阶段引入完整日志框架；如需扩展，建议统一在入口层添加 `logging.basicConfig(...)` 并逐步替换 print 调用。
3. 暂不要求结构化日志字段或分级输出，但应保持输出内容可被 grep 解析（例如固定前缀），便于后续迁移到结构化日志时做自动化替换。