from typing import List

def extract_visible_examples(prompt: str) -> List[str]:
    """提取 HumanEval prompt 中题面可见的示例。"""
    lines = prompt.splitlines()
    examples: List[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        if stripped.startswith(">>>"):
            example_parts = [stripped]
            next_idx = idx + 1
            while next_idx < len(lines):
                continuation = lines[next_idx].rstrip().strip()
                if not continuation:
                    break
                if continuation.startswith(">>>"):
                    break
                if continuation.startswith('"""') or continuation.startswith("'''"):
                    break
                example_parts.append(continuation)
                next_idx += 1
            examples.append("\n".join(example_parts))
            idx = next_idx
            continue

        if "=>" in stripped or "==" in stripped or "➞" in stripped:
            examples.append(stripped)
            idx += 1
            continue

        if stripped.startswith("Input:"):
            example_parts = [stripped]
            next_idx = idx + 1
            while next_idx < len(lines):
                continuation = lines[next_idx].rstrip().strip()
                if not continuation:
                    break
                if continuation.startswith("Example") or continuation.startswith("Input:"):
                    break
                example_parts.append(continuation)
                if continuation.startswith("Output:"):
                    break
                next_idx += 1
            examples.append("\n".join(example_parts))
            idx = next_idx + 1
            continue

        idx += 1
    return examples

def build_final_code_prompt(prompt: str, history: list) -> str:
    """构造最终代码合成提示词，优先锚定题面示例而非讨论中的臆测约束。"""
    history_text = "\n".join(history) if history else "(无历史讨论)"
    visible_examples = extract_visible_examples(prompt)
    examples_text = "\n".join(visible_examples) if visible_examples else "(题面未提供可抽取示例)"
    return f"""请阅读团队讨论历史，补全并实现以下 Python 函数。

【需求与签名】:
{prompt}

【题面可见示例】:
{examples_text}

【团队讨论历史】:
{history_text}

【极其重要的输出要求】:
0. 【第 0 步 强制算法过滤器，优先级高于 1-13 条，不接受例外】：严格按下列 4 小步执行，一步都不能跳：
   第 1 小步：先从【题面可见示例】中把所有 doctest / Input-Output / ➞ / => / == 对全部抄下来，做成一张明确的「输入 → 期望输出」映射表。
   第 2 小步：从【团队讨论历史】中把所有明确提到过的候选算法 / 实现方案全部列出来（例如 A=set() 比较字符种类、B=Counter() 比较字符频次、C=sorted() 比较、D=牛顿法、E=二分法、F=加 raise ValueError、G=不加 raise 等）。
   第 3 小步：对第 2 小步中的**每一个候选方案**，逐个对第 1 小步映射表中的**每一对 输入→期望输出**做一次**具体数值/语义代入计算**（不能只看表面描述）：
      - 若该候选在任何一对上计算出的结果 ≠ 期望输出 → 该候选被**永久淘汰**（哪怕该候选在其余所有对都正确，也必须直接排除，不能再考虑）。
      - 若该候选在**全部**输入对上计算出的结果都 == 期望 → 保留为"全通候选"。
   第 4 小步：根据保留下来的"全通候选"集合作最终决策：
      - 若全通候选 **只剩 1 个** → 直接选择该候选，**完全忽略**讨论中支持任何已淘汰候选的所有文字（哪怕 Reviewer / Architect 反复强调已淘汰的方案多少次，你也只选这一个全通方案）。
      - 若全通候选 ≥ 2 个 → 再按下面规则 11 的多数派 / 仲裁优先级裁决。
      - 若全通候选为 0 个 → 选择通过示例最多的那个候选。
1. 仅输出最终可执行的 Python 代码实现。
2. 必须包含完整的函数定义并直接可被调用。
3. 不要包含任何 markdown 说明文字或解说，直接写代码。
4. 若团队讨论存在冲突，以题面函数签名、docstring 示例和隐含测试语义为最高优先级，不要臆造额外约束。
5. 优先选择最直接、最贴合示例的正确实现，避免为了“泛化”而偏离题目真实语义。
6. 若【题面可见示例】存在，它们是最终代码必须优先满足的显式约束。
7. 若团队讨论与题面自然语言说明、题面示例或题面运算语义（如优先级、边界条件、tie-break 规则）冲突，忽略讨论中的相反结论，以题面为最终裁决。
8. 不要添加题面和示例均未明确要求的防御性分支或额外边界校验（例如对输入范围、空值、负数、符号位、输入数组长度等做出题面没有显式要求的判断或 raise）；题面没有写应该抛异常时永远不要写 raise 分支，哪怕 Note 里只是陈述长度关系也不要做显式校验。
9. 永远不要因为“我认为示例有误”或“我认为示例需要修正”就推翻题面可见示例或示例对应的正确语义；题面和可见示例是最终裁决。
10. 若 Reviewer 明确给出“形如 输入X → 输出Y”的单点断言（例如“[1, 1, 2] 的次小值应为 1”或“foo(3,4) 必须返回 9”等），请在最终实现中显式写出能够满足该断言的判断/分支/算法，不能仅停留在口头讨论。
11. 当讨论里同时出现多种冲突实现/公式时，按以下顺序选最终方案：
    a) 先统计每种实现/公式的出现次数，有明确多数派（出现次数 ≥ 其他方案之和）时选多数派；
    b) 若打平，优先选 Architect 首次明确支持且 Reviewer 没有举出反例的方案；
    c) 任何被 Reviewer 用具体反例明确否决的方案（例如“该简化公式会导致 n=5 时返回 10，但实际仅 1 个有效三元组”）无论出现多少次，必须排除。
12. 当出现边界条件争议（例如 `>` vs `>=`、`<` vs `<=`、是否 raise、是否处理负数）时，严格按以下优先级裁决：
    a) Reviewer 给出的具体反例（如“shift = 位数时应返回原字符串，而不是反转”）最高；
    b) 若无反例，严格照抄题面文本里的运算符用词，题面写 `>` 就用 `>`，写 `>=` 就用 `>=`，绝不自己替换；
    c) 题面没提到应 raise ValueError 时，永远不要写 raise 分支（哪怕 Reviewer 建议加也要忽略）。
13. 不要输出报错文本、占位符或解释；若有不确定，仍返回你认为最可能通过测试的完整实现。"""
