# 冻结反馈输出增强重评 Implementation Plan

> **执行方式：** 子技能`superpowers:executing-plans`未安装，按用户继续授权内联执行，不创建子代理，不自动提交。

**目标：** 在不重新生成或修复输出的前提下，核查原反馈实验的精度天花板能否经受发布方增强测试。

**架构：** 冻结发布方HF修订及核验的LFS SHA-256，在独立解释器中验收全164题参考实现；只有全部通过才允许重评原128份最终输出。新产物绑定旧协议、report、raw和analysis；两种评分分别保留，不称独立新数据或确认性实验。

**技术栈：** Python、NumPy、显式加入依赖目录的`python -I -S`隔离进程、不可覆盖JSON、任务bootstrap与精确方向检验。

## 任务1：独立增强评分

**文件：** 新增`src/isolated_plus_evaluation.py`及`tests/test_isolated_plus_evaluation.py`。

- [x] 先写独立评分失败测试：

```python
def test_wrong_code_fails_expanded_checks():
    task = {"entry_point": "f", "test": "def check(candidate):\n    assert candidate(3) == 6"}
    assert score_plus("def f(x): return x", task)["passed"] is False
```

- [x] 执行`pytest -q tests/test_isolated_plus_evaluation.py`，确认缺模块失败。
- [x] 实现`score_plus(output, task, seed=20261029, timeout=30)`及`plus_protocol()`；两次独立解释器，分类断言/异常/超时，不一致停机，不继承凭证，代码围栏提取不修复；`-I -S`显式绑定NumPy目录，不启用site初始化。
- [x] 验证正确/错误代码、NumPy容差、凭证隔离、超时、非协议退出、两次不一致、运行环境绑定。

## 任务2：冻结与来源验收

**文件：** 新增`experiments/rescore_feedback_plus.py`及`tests/test_feedback_plus_rescoring.py`。

- [x] 冻结发布方修订`d32357cf319e50e9c8d8dab5ea876c72b0fd321b`及LFS SHA-256 `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`；原样数据、来源及许可声明独立保存。
- [x] 实现`prepare_plus(source_path, output_dir)`，完整164个唯一ID、`check(candidate)`语法、任务入口一致、参考按`prompt + canonical_solution`拼接，冻结实现与数据hash。
- [x] 全164参考各检查两次；163通过，32的导出oracle运行异常。预声明守卫阻止候选评分，不删题、不修复测试。新增`audit_plus_reference_suite.py`只完成源审计，完整失败记录保留。
- [x] 实现并用合成测试验证`rescore_feedback_plus(output_dir)`的旧流水线恢复、原标签保留、独立新评分及内容hash；不改旧证据或以评分选优。
- [ ] **真实128份候选重评未执行：** 全库参考验收失败，不能放行。需另行取得并验收正确来源，不事后放宽守卫。
- [x] 验证checksum漂移、评分材料与生成分离、raw变化、缓存不重算、内容篡改和中断恢复。

## 任务3：统计与交付

**文件：** 生成独立`manifest.json`、`source_audit.json`、`acceptance.json`、`README.md`及`report.html`；完整通过的`reference_acceptance.json`及候选`analysis.json`没有生成。

- [x] 统计代码沿用每题两输出均值、VIB减随机、10000次任务bootstrap与方向检验，合成测试通过；真实增强统计未发布。
- [x] 全量472项通过后执行以下命令，32处参考失败按协议停止：

```bash
python -m experiments.rescore_feedback_plus \
  --source /tmp/vib-evalplus-package/humanevalplus-test.jsonl \
  --output-dir results/feedback_plus_rescore_20261029
```

- [x] 交付来源审计报告，原100%保留、增强成绩标“未执行”，不填0或100%；新模型请求0，旧成本不改。明确HF导出来源与整体超时不是官方CLI语义。
- [x] 禁止本地评分调用的缓存复用通过；旧证据前后hash一致，候选评分目录和analysis均未生成。报告只给实测本地时间，不外推性能。
- [x] 最终全量480项通过（145.07s），专项25项通过。离线1280×720与390×844布局通过，外部请求及页面错误0，未执行标识清晰。候选重评项仍未执行，不标为完成。
