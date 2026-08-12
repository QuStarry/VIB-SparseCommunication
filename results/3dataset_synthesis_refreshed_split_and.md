# Multi-Benchmark Analysis

Source: `results/3dataset_synthesis_refreshed_split_and.json`

## 总体判断

- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。
- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。
- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。

## Pareto 概览 (Msg Compression vs Pass@1 Δ)

横轴是该数据集的消息压缩率，纵轴是 VIB 相对于 Full 的 Pass@1 变化。

| Dataset | Total | Full Pass@1 | VIB Pass@1 | Δ Pass@1 (pp) | Msg Compress | Token Saving | McNemar p |
|---|---|---|---|---|---|---|---|
| humaneval | 164 | 100.0% | 100.0% | +0.00 | 21.14% | 3.88% | 1.0 |
| mbpp | 257 | 90.66% | 91.44% | +0.78 | -0.12% | -0.05% | 0.6875 |
| gsm8k | 1319 | 95.3% | 95.22% | -0.08 | 30.49% | 4.96% | 1.0 |

## 数据集内部分组拆分（Early-Stopped vs Multi-Round）

| Dataset | Group | Count | Full Pass@1 | VIB Pass@1 | Δ (pp) | Msg Compress | Token Saving |
|---|---|---|---|---|---|---|---|
| mbpp | Early-Stopped | 225 (87.55%) | 99.11% | 98.67% | -0.44 | 0.0% | 0.06% |
| mbpp | Multi-Round | 32 (12.45%) | 31.25% | 40.62% | +9.37 | -0.57% | -0.46% |

## Trace 可用性

- `humaneval`: 同次运行 trace 覆盖率 `100.00%`
- `mbpp`: 同次运行 trace 覆盖率 `100.00%`
- `gsm8k`: 同次运行 trace 覆盖率 `100.00%`

## 数据集逐项分析

### humaneval


- `Pass@1` 变化: `+0.00 pp`
- 配对胜负: `VIB only = 0`，`Full only = 0`，净增益 `+0` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `164`，双方都失败: `0`
- 消息不变占比: `33.54%`，平均每题少发消息: `1.2683`
- 平均每题少用 Token: `143.0122`
- Token 分解: `prompt saving = 4.50%`，`completion saving = -0.52%`

### mbpp


- `Pass@1` 变化: `+0.78 pp`
- 配对胜负: `VIB only = 4`，`Full only = 2`，净增益 `+2` 题
- 配对显著性: `McNemar p = 0.6875`
- 双方都成功: `231`，双方都失败: `20`
- 消息不变占比: `95.33%`，平均每题少发消息: `-0.0039`
- 平均每题少用 Token: `-1.07`
- Token 分解: `prompt saving = 0.02%`，`completion saving = -0.60%`
- `Full` 优势样例: `MBPP/306, MBPP/408`
- `VIB` 优势样例: `MBPP/101, MBPP/260, MBPP/285, MBPP/462`
- 同次 trace 的 `Full only` 样例:
  - `MBPP/306`: `full_msgs=3`，`vib_msgs=3`，`full_out=def max_sum_increasing_subseq(arr, n, i, k):
    dp = [0] * n
    for j in range(n):
        dp[j] = arr[j]
        for m in range(j):
            if arr[m] < arr[j]:
                dp[j] = max(dp[j], dp[m] + arr[j])
    result = arr[k]
    for j in range(i + 1):
        if arr[j] < arr[k]:
            result = max(result, dp[j] + arr[k])
    return result`，`vib_out=def max_sum_increasing_subseq(arr, n, i, k):
    dp = [0] * n
    for j in range(n):
        dp[j] = arr[j]
        for m in range(j):
            if arr[m] < arr[j]:
                dp[j] = max(dp[j], dp[m] + arr[j])
    result = arr[k]
    for j in range(i):
        if arr[j] < arr[k]:
            result = max(result, dp[j] + arr[k])
    return result`
  - `MBPP/408`: `full_msgs=3`，`vib_msgs=6`，`full_out=def k_smallest_pairs(nums1, nums2, k):
    import heapq
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
    return result`，`vib_out=def k_smallest_pairs(nums1, nums2, k):
    if not nums1 or not nums2 or k <= 0:
        return []
    if k >= len(nums1) * len(nums2):
        pairs = [[a, b] for a in nums1 for b in nums2]
        pairs.sort(key=lambda x: (x[0] + x[1], x[0], x[1]))
        return pairs
    import heapq
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
    count = 2
    s0, s1 = 1, 1
    i = 2
    while True:
        s2 = 2 * s1 + s0
        if is_prime(s2):
            count += 1
            if count == n:
                return s2
        s0, s1 = s1, s2
        i += 1

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
        a, b = b, c
        if is_prime(c):
            count += 1
            if count == n:
                return c

def is_prime(num):
    if num < 2:
        return False
    if num % 2 == 0:
        return num == 2
    if num % 3 == 0:
        return num == 3
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
