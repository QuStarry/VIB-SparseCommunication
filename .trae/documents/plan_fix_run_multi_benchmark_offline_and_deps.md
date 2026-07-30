## Summary

修复 `experiments/run_multi_benchmark.py` 在加载 MBPP/GSM8K 时被强制开启 HuggingFace 离线模式导致 `OfflineModeIsEnabled` 报错的问题；同时补齐依赖安装与运行指引，并让“离线/在线”行为可通过 CLI 显式控制。

## Current State Analysis

### 复现现象

- 运行 `python3 experiments/run_multi_benchmark.py --datasets humaneval mbpp gsm8k` 时：
  - MBPP/GSM8K 加载失败，报：`Couldn't reach 'mbpp' on the Hub (OfflineModeIsEnabled)` / `Couldn't reach 'gsm8k' on the Hub (OfflineModeIsEnabled)`
  - 脚本随后提示“请安装 datasets 库”，但该提示对“离线模式导致的失败”并不准确。

### 根因定位（已在仓库内确认）

- `run_multi_benchmark.py` 在 import 之前强制设置离线环境变量：
  - `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
  - 位置：[run_multi_benchmark.py:L12-L16](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py#L12-L16)
- MBPP/GSM8K 是通过 HuggingFace `datasets` 从 Hub 下载/读取：
  - MBPP：[run_multi_benchmark.py:L106-L136](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py#L106-L136)
  - GSM8K：[run_multi_benchmark.py:L139-L167](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py#L139-L167)
- 由于被强制离线，即使安装了 `datasets`，也会因无法访问 Hub 而失败。
- `requirements.txt` 已包含 `datasets>=3.6.0`：
  - [requirements.txt:L1-L13](file:///Users/starry/Development/Code/VIB-SparseCommunication/requirements.txt#L1-L13)

## Proposed Changes

### 1) 让离线模式变为显式可控（默认离线）

**目标**
- 默认保持离线（避免每题/每次运行都发生 Hub 校验或网络抖动带来的不稳定）。
- 当需要首次下载/更新 MBPP/GSM8K 缓存时，通过 `--online` 显式允许联网拉取。

**具体改动**
- 文件：[run_multi_benchmark.py](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py)
  - 删除顶部硬编码：
    - `os.environ["HF_HUB_OFFLINE"] = "1"`
    - `os.environ["TRANSFORMERS_OFFLINE"] = "1"`
  - 在 CLI 增加参数：
    - `--online`（bool，默认 False）
  - 在 `main()` 解析参数后、执行数据集加载前计算：
    - `offline = (not args.online)`
  - 统一由脚本在 `main()` 中设置离线相关变量（避免“先 import datasets 再设置变量”的时序问题）：
    - 若 `offline=True`：设置
      - `HF_HUB_OFFLINE=1`
      - `HF_DATASETS_OFFLINE=1`
      - （可选）`TRANSFORMERS_OFFLINE=1`
    - 若 `offline=False`：显式清理/覆盖为
      - `HF_HUB_OFFLINE=0`
      - `HF_DATASETS_OFFLINE=0`
      - `TRANSFORMERS_OFFLINE=0`

### 2) 修正 MBPP/GSM8K 的异常处理与提示文案

**目标**
- 区分“没装 datasets”和“离线/网络不可达”两类失败原因。

**具体改动**
- 文件：[run_multi_benchmark.py](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py)
  - MBPPAdapter.load / GSM8KAdapter.load：
    - 对 `ImportError`：提示 `pip install -r requirements.txt` 或 `pip install datasets`
    - 对包含 `OfflineModeIsEnabled` 的异常信息：提示“关闭离线模式或先预下载缓存数据集”
    - 其余网络相关异常：提示检查网络/代理/HF 访问

### 3) 让 HumanEval 的下载行为与 offline 一致（避免隐式联网）

**目标**
- 当用户指定 `--offline` 时，HumanEval 若本地缺文件应直接给出明确错误，而不是尝试联网下载。

**具体改动**
- 文件：[run_multi_benchmark.py](file:///Users/starry/Development/Code/VIB-SparseCommunication/experiments/run_multi_benchmark.py)
  - `HumanEvalAdapter.load()`：
    - `--offline` 时如果 `HumanEval.jsonl.gz` 不存在：提示用户手动放置文件（路径保持不变），并返回空列表/抛出受控异常（优先返回空列表保持现有 harness “Skipped” 行为）
    - 非离线时保持现状：自动下载

实现细节建议：将 offline 作为参数透传给 adapter（例如给 `BaseDatasetAdapter.load()` 增加 `offline: bool = False`），或在 adapter 内读取环境变量 `HF_HUB_OFFLINE/HF_DATASETS_OFFLINE`（更少改动，但可读性略弱）。

### 4) 补齐运行与依赖安装指引（不改核心流程）

**目标**
- 给出“安装依赖 + 在线跑 + 离线预缓存”的标准操作路径，减少误判。

**具体改动**
- 在 `run_multi_benchmark.py` 的 argparse help 或启动时打印一段简短说明：
  - 安装：`pip install -r requirements.txt`
  - 默认离线运行：若本地没有 MBPP/GSM8K 缓存会提示你先 `--online` 跑一次
  - 在线运行：加 `--online`（脚本会自动将 HF 离线变量覆盖为 0）

## Assumptions & Decisions

- 决策：默认行为保持离线；首次拉取/更新数据集缓存需要显式 `--online`。
- 决策：不引入新的第三方库；继续使用现有 `datasets` 生态。
- 假设：运行环境允许安装 Python 依赖（pip/conda 均可），并在需要时具备联网下载数据集能力。

## Verification Steps

### 本地依赖与语法验证
- `python3 -m py_compile experiments/run_multi_benchmark.py`

### 依赖安装验证
- `pip install -r requirements.txt`
- `python3 -c "import datasets; print(datasets.__version__)"`

### 功能验证（在线）
- 用 `--online` 运行（脚本自动覆盖 HF 离线变量为 0）：
  - `python3 experiments/run_multi_benchmark.py --datasets mbpp gsm8k --limit 1 --online`

### 功能验证（离线）
- 先在线预缓存一次（可用上面的 `--online` 跑一次，或用 Python one-liner 下载缓存）：
  - `python3 -c "from datasets import load_dataset; load_dataset('mbpp','sanitized',split='test'); load_dataset('gsm8k','main',split='test')"`
- 默认离线运行（不加 `--online`）：
  - `python3 experiments/run_multi_benchmark.py --datasets mbpp gsm8k --limit 1`

### 回归验证
- 跑现有测试：`pytest -q`
