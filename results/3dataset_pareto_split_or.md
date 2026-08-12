# Multi-Benchmark Analysis

Source: `results/3dataset_cli_frozen_only_split_or.json`

## 总体判断

- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。
- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。
- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。

## Pareto 概览 (Msg Compression vs Pass@1 Δ)

横轴是该数据集的消息压缩率，纵轴是 VIB 相对于 Full 的 Pass@1 变化。

| Dataset | Total | Full Pass@1 | VIB Pass@1 | Δ Pass@1 (pp) | Msg Compress | Token Saving | McNemar p |
|---|---|---|---|---|---|---|---|
| humaneval | 164 | 89.63% | 89.63% | +0.00 | 21.14% | 3.88% | 1.0 |
| mbpp | 257 | 91.83% | 93.39% | +1.56 | -0.12% | -0.05% | 0.21875 |
| gsm8k | 1319 | 95.3% | 95.22% | -0.08 | 30.49% | 4.96% | 1.0 |

## 数据集内部分组拆分（Early-Stopped vs Multi-Round）

| Dataset | Group | Count | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Saving |
|---|---|---|---|---|---|---|---|
| mbpp | Early-Stopped | 236 (91.83%) | 99.15% | 99.58% | +0.43 | -0.41% | 0.05% |
| mbpp | Multi-Round | 21 (8.17%) | 9.52% | 23.81% | +14.29 | 1.59% | -0.63% |

## Trace 可用性

- `humaneval`: 同次运行 trace 覆盖率 `100.00%`
- `mbpp`: 同次运行 trace 覆盖率 `100.00%`
- `gsm8k`: 同次运行 trace 覆盖率 `100.00%`

## 数据集逐项分析

### humaneval


- `Pass@1` 变化: `+0.00 pp`
- 配对胜负: `VIB only = 8`，`Full only = 8`，净增益 `+0` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `139`，双方都失败: `9`
- 消息不变占比: `33.54%`，平均每题少发消息: `1.2683`
- 平均每题少用 Token: `143.0122`
- Token 分解: `prompt saving = 4.50%`，`completion saving = -0.52%`
- `Full` 优势样例: `HumanEval/54, HumanEval/68, HumanEval/75, HumanEval/81, HumanEval/83, HumanEval/130, HumanEval/158, HumanEval/160`
- `VIB` 优势样例: `HumanEval/1, HumanEval/32, HumanEval/93, HumanEval/116, HumanEval/119, HumanEval/129, HumanEval/134, HumanEval/146`
- 同次 trace 的 `Full only` 样例:
  - `HumanEval/54`: `full_msgs=6`，`vib_msgs=6`，`full_out=def same_chars(s0: str, s1: str):
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
    return set(s0) == set(s1)`，`vib_out=def same_chars(s0: str, s1: str):
    from collections import Counter
    return Counter(s0) == Counter(s1)`
  - `HumanEval/68`: `full_msgs=6`，`vib_msgs=4`，`full_out=def pluck(arr):
    min_val = float('inf')
    min_idx = -1
    for i, v in enumerate(arr):
        if v % 2 == 0 and v < min_val:
            min_val = v
            min_idx = i
    if min_idx == -1:
        return []
    return [min_val, min_idx]`，`vib_out=def pluck(arr):
    return min(((v, i) for i, v in enumerate(arr) if v % 2 == 0), default=[])`
  - `HumanEval/75`: `full_msgs=6`，`vib_msgs=6`，`full_out=def is_multiply_prime(a):
    if a < 8:
        return False
    primes = []
    for num in range(2, 100):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    for p in primes:
        for q in primes:
            for r in primes:
                if p * q * r == a:
                    return True
    return False`，`vib_out=def is_multiply_prime(a):
    primes = []
    for num in range(2, 100):
        is_prime = True
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    products = set()
    for i in primes:
        for j in primes:
            for k in primes:
                product = i * j * k
                if product < 100:
                    products.add(product)
    
    return a in products`
- 同次 trace 的 `VIB only` 样例:
  - `HumanEval/1`: `full_msgs=6`，`vib_msgs=2`，`full_out=from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    depth = 0
    start = 0
    for i, char in enumerate(paren_string):
        if char == ' ':
            continue
        if char == '(':
            if depth == 0:
                start = i
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                result.append(paren_string[start:i + 1])
    return result`，`vib_out=from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    depth = 0
    start = -1
    for i, ch in enumerate(paren_string):
        if ch == ' ':
            continue
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                result.append(paren_string[start:i+1].replace(' ', ''))
    return result`
  - `HumanEval/32`: `full_msgs=6`，`vib_msgs=5`，`full_out=import math


def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees
    a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    n = len(xs) - 1
    x = 0.0
    for _ in range(100):
        f = poly(xs, x)
        if abs(f) < 1e-10:
            return x
        df = sum([i * coeff * math.pow(x, i - 1) for i, coeff in enumerate(xs) if i > 0])
        if abs(df) < 1e-10:
            break
        x = x - f / df
    return x`，`vib_out=import math


def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees
    a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    if len(xs) % 2 != 0:
        return None
    if xs[-1] == 0:
        return None

    def derivative(xs):
        return [i * coeff for i, coeff in enumerate(xs)][1:]

    def f(x):
        return poly(xs, x)

    def f_prime(x):
        return poly(derivative(xs), x)

    x = 0.0
    for _ in range(1000):
        fx = f(x)
        if abs(fx) < 1e-10:
            return x
        fpx = f_prime(x)
        if fpx == 0:
            break
        x = x - fx / fpx
    return x`
  - `HumanEval/93`: `full_msgs=6`，`vib_msgs=3`，`full_out=def encode(message):
    vowel_map = {
        'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
        'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'
    }
    result = []
    for ch in message:
        swapped = ch.swapcase()
        if ch in vowel_map:
            result.append(vowel_map[ch])
        else:
            result.append(swapped)
    return ''.join(result)`，`vib_out=def encode(message):
    vowels = {'a': 'c', 'e': 'g', 'i': 'k', 'o': 'q', 'u': 'w',
              'A': 'C', 'E': 'G', 'I': 'K', 'O': 'Q', 'U': 'W'}
    result = []
    for ch in message:
        swapped = ch.swapcase()
        if swapped in vowels:
            result.append(vowels[swapped])
        else:
            result.append(swapped)
    return ''.join(result)`

### mbpp


- `Pass@1` 变化: `+1.56 pp`
- 配对胜负: `VIB only = 5`，`Full only = 1`，净增益 `+4` 题
- 配对显著性: `McNemar p = 0.21875`
- 双方都成功: `235`，双方都失败: `16`
- 消息不变占比: `95.33%`，平均每题少发消息: `-0.0039`
- 平均每题少用 Token: `-1.07`
- Token 分解: `prompt saving = 0.02%`，`completion saving = -0.60%`
- `Full` 优势样例: `MBPP/408`
- `VIB` 优势样例: `MBPP/101, MBPP/260, MBPP/285, MBPP/304, MBPP/462`
- 同次 trace 的 `Full only` 样例:
  - `MBPP/408`: `full_msgs=3`，`vib_msgs=6`，`full_out=import heapq

def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2:
        return []
    nums1.sort()
    nums2.sort()
    result = []
    heap = []
    visited = set()
    for i in range(len(nums1)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
        visited.add((i, 0))
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))
    return result`，`vib_out=import heapq

def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k <= 0:
        return []
    
    if k >= len(nums1) * len(nums2):
        result = []
        for i in nums1:
            for j in nums2:
                result.append([i, j])
        result.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))
        return result
    
    heap = []
    for i in range(len(nums1)):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
    
    result = []
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
        
        if j == 0 and i + 1 < len(nums1):
            heapq.heappush(heap, (nums1[i + 1] + nums2[0], i + 1, 0))
    
    return result`
- 同次 trace 的 `VIB only` 样例:
  - `MBPP/101`: `full_msgs=6`，`vib_msgs=3`，`full_out=def kth_element(arr, k):
    return sorted(arr)[k - 1]`，`vib_out=def kth_element(arr, k):
    return arr[k - 1]`
  - `MBPP/260`: `full_msgs=6`，`vib_msgs=6`，`full_out=def newman_prime(n):
    if n == 1:
        return 2
    if n == 2:
        return 3
    
    s0, s1 = 1, 1
    count = 2
    idx = 2
    
    while True:
        s2 = 2 * s1 + s0
        if is_prime(s2):
            count += 1
            if count == n:
                return s2
        s0, s1 = s1, s2
        idx += 1

def is_prime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True`，`vib_out=def newman_prime(n):
    if n == 1:
        return 3
    if n == 2:
        return 7
    count = 2
    a, b = 1, 3
    while True:
        c = 2 * b + a
        if is_prime(c):
            count += 1
            if count == n:
                return c
        a, b = b, c

def is_prime(num):
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
    return True`
  - `MBPP/285`: `full_msgs=6`，`vib_msgs=6`，`full_out=import re

def text_match_two_three(text):
    return bool(re.search(r'ab{2,3}(?!b)', text))`，`vib_out=import re

def text_match_two_three(text):
    return bool(re.search(r'ab{2,3}', text))`

### gsm8k


- `Pass@1` 变化: `-0.08 pp`
- 配对胜负: `VIB only = 13`，`Full only = 14`，净增益 `-1` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `1243`，双方都失败: `49`
- 消息不变占比: `16.98%`，平均每题少发消息: `1.8294`
- 平均每题少用 Token: `115.2062`
- Token 分解: `prompt saving = 5.89%`，`completion saving = -0.16%`
- `Full` 优势样例: `GSM8K/37, GSM8K/40, GSM8K/508, GSM8K/526, GSM8K/564, GSM8K/641, GSM8K/754, GSM8K/796, GSM8K/797, GSM8K/942`
- `VIB` 优势样例: `GSM8K/85, GSM8K/89, GSM8K/120, GSM8K/340, GSM8K/368, GSM8K/652, GSM8K/672, GSM8K/675, GSM8K/753, GSM8K/857`
- 同次 trace 的 `Full only` 样例:
  - `GSM8K/37`: `full_msgs=6`，`vib_msgs=3`，`full_out=答案: 2`，`vib_out=答案: 0`
  - `GSM8K/40`: `full_msgs=6`，`vib_msgs=5`，`full_out=答案: 8`，`vib_out=答案: 12`
  - `GSM8K/508`: `full_msgs=6`，`vib_msgs=3`，`full_out=答案: 120`，`vib_out=答案: 75`
- 同次 trace 的 `VIB only` 样例:
  - `GSM8K/85`: `full_msgs=6`，`vib_msgs=6`，`full_out=答案: 48`，`vib_out=答案: 44`
  - `GSM8K/89`: `full_msgs=6`，`vib_msgs=3`，`full_out=答案: 18`，`vib_out=答案: 24`
  - `GSM8K/120`: `full_msgs=6`，`vib_msgs=6`，`full_out=答案: 257`，`vib_out=答案: 240`

## 对先前讨论的修正

- HumanEval 和 MBPP 的提升需要谨慎表述。如果净增益只有 1 题，就更适合写成“结果持平、略优于 Full”，而不是“显著提升”。
- GSM8K 的下降也不能只看 `-0.99pp`，还要看它对应的是多少条配对逆转样本，以及这些样本是否集中在某些推理模式上。
- “消息压缩率远高于 Token 节省率”这个判断方向通常是对的，但需要 `prompt/completion` 拆分数据支撑，不能只凭经验解释。
- MBPP 压缩率低不一定只是“代码任务信息密度高”，还可能与当前实验配置有关，例如 `max_steps=2`、三智能体上限接近固定 6 条消息、以及消息模板较短。

## 建议的下一步

- 优先分析 `full_only`/`vib_only` 的逐题模式，尤其是 GSM8K 的 `full_only` 逆转样本。
- 后续实验固定记录 `prompt/completion` 拆分，避免再对 Token 节省原因做不可证伪的推断。
- 若要引入更强的自适应机制，先验证当前客户端/模型是否真的支持 `logprobs`、低开销前缀探针和可复用 KV cache；否则相关讨论仍属于架构设想，而不是可直接落地的实验结论。
