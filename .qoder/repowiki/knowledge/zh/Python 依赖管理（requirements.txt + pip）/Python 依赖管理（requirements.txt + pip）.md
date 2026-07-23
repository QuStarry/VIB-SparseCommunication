---
kind: dependency_management
name: Python 依赖管理（requirements.txt + pip）
category: dependency_management
scope:
    - '**'
source_files:
    - requirements.txt
---

本项目采用最简化的 Python 依赖管理方案，仅通过根目录的 `requirements.txt` 声明第三方依赖，未使用任何现代包管理器（如 Poetry、PDM、uv、pipenv）。

**核心机制**
- 单一清单文件：`requirements.txt`，按功能分组注释（Core dependencies / Testing / Optional: for visualization），使用 `>=` 指定最低版本。
- 无锁定文件：仓库中不存在 `requirements.lock`、`poetry.lock`、`uv.lock` 等锁文件，也未在 `.gitignore` 中显式排除它们（`.gitignore` 中的相关条目来自模板默认内容）。
- 无虚拟环境配置：未发现 `pyproject.toml`、`setup.py`、`Pipfile`、`poetry.lock`、`uv.lock`、`conda.yml` 等任一现代依赖管理配置文件。
- 无私有源/镜像：`requirements.txt` 中未出现 `-i`、`--index-url`、`--extra-index-url` 等 pip 选项，所有包均来自 PyPI 官方源。
- 无 vendoring：未发现 `vendor/`、`third_party/` 或内嵌的源码副本。

**关键依赖**
- 核心：`numpy>=1.24.0`、`sentence-transformers>=2.2.0`、`langgraph>=0.2.0`、`langchain-core>=0.3.0`
- 测试：`pytest>=7.0.0`
- 可选可视化：`matplotlib>=3.7.0`
- `sentence-transformers` 在 `src/vib_router.py` 中以延迟 import 方式加载，安装失败时给出明确的 `pip install sentence-transformers` 提示。

**开发者约定**
- 新增依赖直接追加到 `requirements.txt` 对应分组下，保持 `package>=version` 格式。
- 由于缺少锁文件，不同环境的依赖解析结果可能不一致，建议在本地开发时使用虚拟环境并自行记录已验证的版本组合。
- 可选依赖（如 matplotlib）通过运行时检测/延迟导入避免强制安装。