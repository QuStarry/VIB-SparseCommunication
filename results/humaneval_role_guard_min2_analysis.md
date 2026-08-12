# Multi-Benchmark Analysis

Source: `results/humaneval_role_guard_min2.json`

## 总体判断

- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。
- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。
- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。

## Trace 可用性

- `humaneval`: 同次运行 trace 覆盖率 `100.00%`

## 数据集逐项分析

### humaneval


- `Pass@1` 变化: `+3.05 pp`
- 配对胜负: `VIB only = 10`，`Full only = 5`，净增益 `+5` 题
- 配对显著性: `McNemar p = 0.301758`
- 双方都成功: `139`，双方都失败: `10`
- 消息不变占比: `29.27%`，平均每题少发消息: `1.2805`
- 平均每题少用 Token: `170.5549`
- Token 分解: `prompt saving = 5.93%`，`completion saving = 3.95%`
- `Full` 优势样例: `HumanEval/54, HumanEval/65, HumanEval/130, HumanEval/135, HumanEval/144`
- `VIB` 优势样例: `HumanEval/1, HumanEval/39, HumanEval/59, HumanEval/81, HumanEval/83, HumanEval/108, HumanEval/116, HumanEval/129, HumanEval/134, HumanEval/136`
- 同次 trace 的 `Full only` 样例:
  - `HumanEval/54`: `full_msgs=6`，`vib_msgs=4`，`full_out=def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
    return set(s0) == set(s1)`，`vib_out=from collections import Counter

def same_chars(s0: str, s1: str):
    return Counter(s0) == Counter(s1)`
  - `HumanEval/65`: `full_msgs=6`，`vib_msgs=6`，`full_out=def circular_shift(x, shift):
    s = str(x)
    n = len(s)
    if shift <= 0:
        return s
    if shift > n:
        return s[::-1]
    shift = shift % n
    if shift == 0:
        return s
    return s[-shift:] + s[:-shift]`，`vib_out=def circular_shift(x, shift):
    s = str(x)
    if s == "0":
        return "0"
    n = len(s)
    shift = shift % n
    if shift == 0:
        return s
    return s[-shift:] + s[:-shift]`
  - `HumanEval/130`: `full_msgs=6`，`vib_msgs=6`，`full_out=def tri(n):
    if n == 0:
        return [1]
    if n == 1:
        return [1, 3]
    
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 3
    
    # 先计算所有偶数项
    for i in range(2, n + 1):
        if i % 2 == 0:
            dp[i] = 1 + i // 2
    
    # 再计算奇数项
    for i in range(3, n + 1, 2):
        if i + 1 <= n:
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i + 1]
        else:
            # 当 i+1 超出范围时，临时计算 tri(i+1)
            temp = 1 + (i + 1) // 2
            dp[i] = dp[i - 1] + dp[i - 2] + temp
    
    return dp[:n + 1]`，`vib_out=def tri(n):
    if n == 0:
        return [1]
    res = [0] * (n + 1)
    res[0] = 1
    res[1] = 3
    # 先计算所有偶数项
    for i in range(2, n + 1, 2):
        res[i] = 1 + i // 2
    # 再计算奇数项
    for i in range(3, n + 1, 2):
        res[i] = res[i - 1] + res[i - 2] + res[i + 1]
    return res`
- 同次 trace 的 `VIB only` 样例:
  - `HumanEval/1`: `full_msgs=6`，`vib_msgs=4`，`full_out=from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    stack = []
    result = []
    start = 0
    for i, char in enumerate(paren_string):
        if char == '(':
            if not stack:
                start = i
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack:
                    result.append(paren_string[start:i+1])
    return result`，`vib_out=from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    if not paren_string or not paren_string.strip():
        return []
    
    result = []
    counter = 0
    start = 0
    
    for i, char in enumerate(paren_string):
        if char == '(':
            counter += 1
        elif char == ')':
            counter -= 1
            if counter == 0:
                result.append(paren_string[start:i+1].replace(' ', ''))
                start = i + 1
        # ignore spaces and other characters
    
    return result`
  - `HumanEval/39`: `full_msgs=6`，`vib_msgs=3`，`full_out=def prime_fib(n: int):
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 2
    
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        if num == 2 or num == 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True
    
    a, b = 0, 1
    count = 0
    index = 0
    while True:
        if index % 3 == 0 and index > 0:
            if is_prime(b):
                count += 1
                if count == n:
                    return b
        a, b = b, a + b
        index += 1`，`vib_out=def prime_fib(n: int):
    if n <= 0:
        raise ValueError("n must be positive")
    
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        if num == 2:
            return True
        if num % 2 == 0:
            return False
        i = 3
        while i * i <= num:
            if num % i == 0:
                return False
            i += 2
        return True
    
    a, b = 1, 2
    count = 0
    while True:
        if is_prime(a):
            count += 1
            if count == n:
                return a
        a, b = b, a + b`
  - `HumanEval/59`: `full_msgs=6`，`vib_msgs=6`，`full_out=def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    # Handle factor 2
    while n % 2 == 0:
        n //= 2
    if n == 1:
        return 2
    
    # Check odd factors from 3 upwards
    i = 3
    while i * i <= n:
        while n % i == 0:
            n //= i
        i += 2
    
    # If n is still greater than 1, it's the largest prime factor
    return n if n > 1 else 2`，`vib_out=def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime."""
    factor = 2
    while factor * factor <= n:
        if n % factor == 0:
            n //= factor
        else:
            factor += 1 if factor == 2 else 2
    return n`

## 对先前讨论的修正

- HumanEval 和 MBPP 的提升需要谨慎表述。如果净增益只有 1 题，就更适合写成“结果持平、略优于 Full”，而不是“显著提升”。
- GSM8K 的下降也不能只看 `-0.99pp`，还要看它对应的是多少条配对逆转样本，以及这些样本是否集中在某些推理模式上。
- “消息压缩率远高于 Token 节省率”这个判断方向通常是对的，但需要 `prompt/completion` 拆分数据支撑，不能只凭经验解释。
- MBPP 压缩率低不一定只是“代码任务信息密度高”，还可能与当前实验配置有关，例如 `max_steps=2`、三智能体上限接近固定 6 条消息、以及消息模板较短。

## 建议的下一步

- 优先分析 `full_only`/`vib_only` 的逐题模式，尤其是 GSM8K 的 `full_only` 逆转样本。
- 后续实验固定记录 `prompt/completion` 拆分，避免再对 Token 节省原因做不可证伪的推断。
- 若要引入更强的自适应机制，先验证当前客户端/模型是否真的支持 `logprobs`、低开销前缀探针和可复用 KV cache；否则相关讨论仍属于架构设想，而不是可直接落地的实验结论。
