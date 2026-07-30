# Multi-Benchmark Analysis

Source: `results/multi_benchmark_limit10_workers4.json`

## 总体判断

- 仅看 `Pass@1` 百分比会放大结论，需要结合配对胜负数、显著性和逐题差分一起看。
- `full_only_count` 表示 `Full 成功但 VIB 失败`；`vib_only_count` 表示 `VIB 成功但 Full 失败`。
- `mcnemar_pvalue` 越小，说明两种设置在同一批任务上的差异越不像随机波动。

## Trace 可用性

- `humaneval`: 同次运行 trace 覆盖率 `100.00%`
- `mbpp`: 同次运行 trace 覆盖率 `100.00%`
- `gsm8k`: 同次运行 trace 覆盖率 `100.00%`

## 数据集逐项分析

### humaneval


- `Pass@1` 变化: `-2.43 pp`
- 配对胜负: `VIB only = 5`，`Full only = 9`，净增益 `-4` 题
- 配对显著性: `McNemar p = 0.42395`
- 双方都成功: `136`，双方都失败: `14`
- 消息不变占比: `21.95%`，平均每题少发消息: `1.9207`
- 平均每题少用 Token: `202.0549`
- Token 分解: `prompt saving = 8.15%`，`completion saving = 0.53%`
- `Full` 优势样例: `HumanEval/39, HumanEval/106, HumanEval/113, HumanEval/121, HumanEval/129, HumanEval/146, HumanEval/147, HumanEval/158, HumanEval/160`
- `VIB` 优势样例: `HumanEval/1, HumanEval/65, HumanEval/75, HumanEval/81, HumanEval/92`
- 同次 trace 的 `Full only` 样例:
  - `HumanEval/39`: `full_msgs=6`，`vib_msgs=3`，`full_out=def prime_fib(n: int):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
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
    
    count = 0
    a, b = 1, 2
    while True:
        if is_prime(b):
            count += 1
            if count == n:
                return b
        a, b = b, a + b`，`vib_out=def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
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

    if n == 1:
        return 2

    count = 1
    a, b = 2, 3
    while True:
        if is_prime(b):
            count += 1
            if count == n:
                return b
        a, b = b, a + b
        if is_prime(a):
            count += 1
            if count == n:
                return a
        a, b = b, a + b`
  - `HumanEval/106`: `full_msgs=6`，`vib_msgs=5`，`full_out=def f(n):
    if n < 1:
        return []
    result = []
    factorial = 1
    sum_ = 0
    for i in range(1, n + 1):
        factorial *= i
        sum_ += i
        if i % 2 == 0:
            result.append(factorial)
        else:
            result.append(sum_)
    return result`，`vib_out=def f(n):
    if n <= 0:
        return []
    result = []
    fact = 1
    for i in range(1, n + 1):
        if i % 2 == 0:
            fact *= i
            result.append(fact)
        else:
            result.append(i * (i + 1) // 2)
    return result`
  - `HumanEval/113`: `full_msgs=6`，`vib_msgs=6`，`full_out=def odd_count(lst):
    result = []
    for s in lst:
        odd_count = sum(1 for ch in s if int(ch) % 2 == 1)
        result.append(f"the number of odd elements {odd_count}n the str{odd_count}ng {odd_count} of the {odd_count}nput.")
    return result`，`vib_out=def odd_count(lst):
    result = []
    for s in lst:
        count = sum(1 for c in s if c in '13579')
        template = "the number of odd elements n the strng  of the nput."
        output = template.replace("n", str(count)).replace(" ", " ", 1)
        output = output.replace("strng", "str" + str(count) + "ng")
        output = output.replace(" of the ", " " + str(count) + " of the ")
        output = output.replace("nput.", str(count) + "nput.")
        result.append(output)
    return result`
- 同次 trace 的 `VIB only` 样例:
  - `HumanEval/1`: `full_msgs=6`，`vib_msgs=4`，`full_out=from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    count = 0
    start = 0
    for i, ch in enumerate(paren_string):
        if ch == ' ':
            continue
        if ch == '(':
            if count == 0:
                start = i
            count += 1
        elif ch == ')':
            count -= 1
            if count == 0:
                result.append(paren_string[start:i+1])
    return result`，`vib_out=from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    paren_string = paren_string.replace(" ", "")
    result = []
    stack = []
    start = 0
    for i, ch in enumerate(paren_string):
        if ch == '(':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == ')':
            stack.pop()
            if not stack:
                result.append(paren_string[start:i+1])
    return result`
  - `HumanEval/65`: `full_msgs=6`，`vib_msgs=6`，`full_out=def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    """
    if x == 0:
        return "0"
    sign = ""
    if x < 0:
        sign = "-"
        x = -x
    s = str(x)
    n = len(s)
    if shift >= n:
        return sign + s[::-1]
    shift = shift % n
    result = s[-shift:] + s[:-shift]
    return sign + result`，`vib_out=def circular_shift(x, shift):
    if x == 0:
        return "0"
    sign = ""
    if x < 0:
        sign = "-"
        x = -x
    s = str(x)
    if shift < 0:
        shift = -shift
        shift = shift % len(s)
        s = s[shift:] + s[:shift]
    elif shift == 0:
        pass
    elif shift > len(s):
        s = s[::-1]
    else:
        shift = shift % len(s)
        s = s[-shift:] + s[:-shift]
    return sign + s`
  - `HumanEval/75`: `full_msgs=6`，`vib_msgs=6`，`full_out=def is_multiply_prime(a):
    if a < 2 or a >= 100:
        return False
    primes = []
    for num in range(2, 100):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    for p1 in primes:
        for p2 in primes:
            for p3 in primes:
                if p1 * p2 * p3 == a:
                    return True
    return False`，`vib_out=def is_multiply_prime(a):
    if a < 8:
        return False
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p1 in primes:
        for p2 in primes:
            for p3 in primes:
                if p1 * p2 * p3 == a:
                    return True
    return False`

### mbpp


- `Pass@1` 变化: `-0.39 pp`
- 配对胜负: `VIB only = 3`，`Full only = 4`，净增益 `-1` 题
- 配对显著性: `McNemar p = 1.0`
- 双方都成功: `227`，双方都失败: `23`
- 消息不变占比: `96.50%`，平均每题少发消息: `0.07`
- 平均每题少用 Token: `13.0973`
- Token 分解: `prompt saving = 0.11%`，`completion saving = 2.64%`
- `Full` 优势样例: `MBPP/160, MBPP/162, MBPP/434, MBPP/450`
- `VIB` 优势样例: `MBPP/100, MBPP/297, MBPP/306`
- 同次 trace 的 `Full only` 样例:
  - `MBPP/160`: `full_msgs=6`，`vib_msgs=6`，`full_out=def find_solution(a, b, n):
    # 扩展欧几里得算法，返回 (g, x, y) 使得 a*x + b*y = g
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        else:
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)

    # 处理 a 或 b 为 0 的边界情况
    if a == 0 and b == 0:
        return None if n != 0 else (0, 0)
    if a == 0:
        if n % b != 0:
            return None
        y = n // b
        # 需要正整数解，这里测试用例要求正整数，但题目只要求整数，我们按正整数调整
        # 但为了通过测试，我们返回一个解，如果 y 是整数即可
        return (0, y)
    if b == 0:
        if n % a != 0:
            return None
        x = n // a
        return (x, 0)

    g, x0, y0 = egcd(a, b)
    if n % g != 0:
        return None

    # 缩放得到特解
    x0 *= n // g
    y0 *= n // g

    # 通解: x = x0 + (b/g) * t, y = y0 - (a/g) * t
    # 调整到正整数解，测试用例中均为正整数，我们尝试找到一组正整数解
    step_x = b // g
    step_y = a // g

    # 尝试调整 t 使得 x 和 y 都为正整数
    # 先尝试 t=0
    if x0 > 0 and y0 > 0:
        return (x0, y0)

    # 尝试调整 t 使 x 和 y 都为正
    # 根据 x > 0 得到 t > -x0/step_x，根据 y > 0 得到 t < y0/step_y
    # 注意 step_x 和 step_y 可能为负，需处理符号
    if step_x > 0:
        t_min = (-x0) // step_x + 1
    else:
        t_min = (-x0) // step_x  # 注意负数除法
        if (-x0) % step_x != 0:
            t_min += 1

    if step_y > 0:
        t_max = (y0 - 1) // step_y
    else:
        t_max = (y0 - 1) // step_y
        if (y0 - 1) % step_y != 0:
            t_max -= 1

    # 尝试在范围内找一个整数 t
    for t in range(t_min, t_max + 1):
        x = x0 + step_x * t
        y = y0 - step_y * t
        if x > 0 and y > 0:
            return (x, y)

    # 如果找不到正整数解，返回 None（但测试用例中应该能找到）
    return None`，`vib_out=def find_solution(a, b, n):
    # 扩展欧几里得算法，返回 (g, x, y) 使得 a*x + b*y = g = gcd(a, b)
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = extended_gcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

    # 处理 a 和 b 均为 0 的情况
    if a == 0 and b == 0:
        if n == 0:
            return (0, 0)  # 任意整数解，这里返回 (0,0)
        else:
            return None

    # 处理 a == 0 的情况
    if a == 0:
        if n % b == 0:
            y = n // b
            return (0, y)
        else:
            return None

    # 处理 b == 0 的情况
    if b == 0:
        if n % a == 0:
            x = n // a
            return (x, 0)
        else:
            return None

    g, x0, y0 = extended_gcd(a, b)
    if n % g != 0:
        return None

    # 缩放得到原方程的一组特解
    scale = n // g
    x0 *= scale
    y0 *= scale

    # 通解: x = x0 + (b/g) * t, y = y0 - (a/g) * t
    # 调整到最小非负整数解（优先保证 x 非负）
    step_x = b // g
    step_y = a // g

    # 调整 x 到最小非负值
    if step_x != 0:
        t = - (x0 // step_x)
        if x0 % step_x != 0 and x0 < 0:
            t += 1
        x0 += t * step_x
        y0 -= t * step_y

    # 如果此时 y 为负，尝试增加 t 使 y 变为非负（同时 x 可能变大）
    if step_y != 0:
        while y0 < 0 and step_y != 0:
            # 增加 t 会使 y 减小（因为 step_y > 0 时 y0 -= step_y * t，t 增加 y 减小）
            # 实际上要使 y 增加，需要减小 t
            # 这里我们直接计算需要的 t 使 y 非负
            if step_y > 0:
                t = (y0 // step_y)  # 因为 y0 负，除以正数得到负数或0
                if y0 % step_y != 0:
                    t -= 1  # 向下取整
                x0 -= t * step_x
                y0 -= t * step_y  # 注意这里 t 是负的，所以 y0 会增加
            else:
                # step_y < 0 的情况
                t = (y0 // step_y)
                if y0 % step_y != 0:
                    t += 1
                x0 -= t * step_x
                y0 -= t * step_y
            break  # 只调整一次

    # 如果调整后 y 仍然为负，尝试增加 t 使 y 非负（循环调整）
    # 更稳健的方法：直接计算使 y 非负的 t 范围
    # 重新计算一次
    x0, y0 = (n // g) * (x0 // scale if scale != 0 else 0), (n // g) * (y0 // scale if scale != 0 else 0)
    # 重新从特解开始调整
    x0 = (n // g) * (x0 // (n // g) if n // g != 0 else 0)
    # 重新计算
    g, x0, y0 = extended_gcd(a, b)
    scale = n // g
    x0 *= scale
    y0 *= scale

    step_x = b // g
    step_y = a // g

    # 调整 x 到最小非负
    if step_x != 0:
        t = - (x0 // step_x)
        if x0 % step_x != 0 and x0 < 0:
            t += 1
        x0 += t * step_x
        y0 -= t * step_y

    # 调整 y 到非负
    if step_y != 0:
        #`
  - `MBPP/162`: `full_msgs=6`，`vib_msgs=6`，`full_out=def sum_series(n):
    return (n + (n - 2 * (n // 2))) * (n // 2 + 1) // 2`，`vib_out=def sum_series(n: int) -> int:
    if n <= 0:
        return 0
    return ((n + 1) // 2) ** 2`
  - `MBPP/434`: `full_msgs=6`，`vib_msgs=6`，`full_out=import re

def text_match_one(s):
    if not isinstance(s, str):
        return False
    return bool(re.search(r'ab+', s))`，`vib_out=import re

def text_match_one(text: str) -> bool:
    return bool(re.fullmatch(r'ab+', text))`
- 同次 trace 的 `VIB only` 样例:
  - `MBPP/100`: `full_msgs=6`，`vib_msgs=6`，`full_out=def next_smallest_palindrome(num):
    if num < 0:
        return 1
    if num < 9:
        return num + 1
    s = str(num)
    n = len(s)
    half = s[:(n + 1) // 2]
    if n % 2 == 0:
        candidate = half + half[::-1]
    else:
        candidate = half + half[-2::-1]
    if int(candidate) > num:
        return int(candidate)
    # 需要进位
    half_int = int(half) + 1
    half_str = str(half_int)
    if len(half_str) > len(half):
        # 进位导致位数增加，如 99 -> 101
        if n % 2 == 0:
            candidate = half_str + half_str[::-1]
        else:
            candidate = half_str + half_str[-2::-1]
    else:
        if n % 2 == 0:
            candidate = half_str + half_str[::-1]
        else:
            candidate = half_str + half_str[-2::-1]
    return int(candidate)`，`vib_out=def next_smallest_palindrome(num):
    if num < 0:
        return 0
    if num < 10:
        return num + 1
    s = str(num)
    n = len(s)
    half = s[:(n + 1) // 2]
    candidate = int(half + half[-1 - (n % 2)::-1])
    if candidate > num:
        return candidate
    half = str(int(half) + 1)
    if len(half) > (n + 1) // 2:
        return 10 ** n + 1
    return int(half + half[-1 - (n % 2)::-1])`
  - `MBPP/297`: `full_msgs=6`，`vib_msgs=6`，`full_out=from collections.abc import Iterable

def flatten_list(nested_list):
    result = []
    stack = [nested_list]
    while stack:
        item = stack.pop()
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            stack.extend(reversed(item))
        else:
            result.append(item)
    result.reverse()
    return result`，`vib_out=def flatten_list(nested_list):
    result = []
    stack = [nested_list]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            result.append(item)
    return result`
  - `MBPP/306`: `full_msgs=6`，`vib_msgs=5`，`full_out=def max_sum_increasing_subseq(arr, n, i, k):
    # 如果 arr[k] <= arr[i]，无法形成递增序列
    if arr[k] <= arr[i]:
        return 0

    # dp[j] 表示以 j 结尾的最大递增子序列和
    dp = [0] * n
    for j in range(n):
        dp[j] = arr[j]
        for p in range(j):
            if arr[p] < arr[j]:
                dp[j] = max(dp[j], dp[p] + arr[j])

    # 寻找从0到i的递增子序列，并包含i，然后连接到包含k的递增子序列
    # 先计算以i结尾的最大递增子序列和（必须包含i）
    best = 0
    # 从0到i中找以i结尾的最大递增子序列和
    for p in range(i + 1):
        if arr[p] < arr[i]:
            best = max(best, dp[p])
    best += arr[i]  # 加上arr[i]本身

    # 现在从i+1到k-1中找中间元素，使得arr[i] < arr[mid] < arr[k]
    # 并计算包含这些中间元素的最大和
    result = 0
    for mid in range(i + 1, k):
        if arr[i] < arr[mid] < arr[k]:
            # 以mid结尾的最大递增子序列和（从i+1到mid）
            mid_sum = arr[mid]
            for p in range(i + 1, mid):
                if arr[p] < arr[mid]:
                    mid_sum = max(mid_sum, dp[p] + arr[mid])
            # 总序列和 = best + mid_sum - arr[i]? 注意best已经包含arr[i]
            # 实际上best是包含i的，mid_sum是包含mid的，但需要确保递增
            # 更简单：直接计算从i到k的递增子序列和
            total = best + mid_sum - arr[i]  # 因为best包含arr[i]，mid_sum包含arr[mid]，但arr[i]和arr[mid]都加了，需要减去重复的arr[i]? 不对，best是到i的，mid_sum是到mid的，它们不重叠
            # 实际上best是到i的，mid_sum是从i+1到mid的，所以直接相加
            total = best + mid_sum - arr[i]  # 减去arr[i]是因为best里已经加了arr[i]，而mid_sum里没有arr[i]，所以不减
            # 重新思考：best = 以i结尾的最大递增子序列和（包含i）
            # mid_sum = 以mid结尾的最大递增子序列和（包含mid），且序列中所有元素都大于arr[i]
            # 那么总序列 = best + (mid_sum - arr[mid])? 不对
            # 正确做法：从0到mid的递增子序列，且包含i和mid，最大和 = dp[mid] 但dp[mid]可能不包含i
            # 所以需要重新计算
            pass

    # 更直接的方法：枚举所有可能的中间元素，计算包含i和k的递增子序列最大和
    # 先计算以i结尾的最大递增子序列和（从0到i）
    dp_i = [0] * (i + 1)
    for j in range(i + 1):
        dp_i[j] = arr[j]
        for p in range(j):
            if arr[p] < arr[j]:
                dp_i[j] = max(dp_i[j], dp_i[p] + arr[j])
    best_i = dp_i[i]

    # 计算从i+1到k-1中每个位置作为中间点的最大和
    max_sum = 0
    for mid in range(i + 1, k):
        if arr[i] < arr[mid] < arr[k]:
            # 计算以mid结尾的最大递增子序列和，且序列中所有元素都大于arr[i]
            dp_mid = arr[mid]
            for p in range(i + 1, mid):
                if arr[p] < arr[mid] and arr[p] > arr[i]:
                    dp_mid = max(dp_mid, dp_mid + arr[p])  # 这里需要重新计算
            # 重新正确计算
            dp_mid = arr[mid]
            for p in range(i + 1, mid):
                if arr[p] < arr[mid] and arr[p] > arr[i]:
                    # 需要以p结尾的递增子序列和，且大于arr[i]
                    # 简单起见，直接计算从i+1到mid的递增子序列最大和（包含mid）
                    pass
    #`，`vib_out=def max_sum_increasing_subseq(arr, n, i, k):
    if not (0 <= i < k < n):
        return 0
    dp = [0] * n
    for idx in range(n):
        dp[idx] = arr[idx]
        for j in range(idx):
            if arr[j] < arr[idx]:
                dp[idx] = max(dp[idx], dp[j] + arr[idx])
    max_sum = 0
    for j in range(i + 1):
        if arr[j] < arr[k]:
            max_sum = max(max_sum, dp[j])
    if max_sum == 0:
        return arr[k] if all(arr[t] >= arr[k] for t in range(i + 1)) else 0
    return max_sum + arr[k]`

### gsm8k


- `Pass@1` 变化: `-0.68 pp`
- 配对胜负: `VIB only = 15`，`Full only = 24`，净增益 `-9` 题
- 配对显著性: `McNemar p = 0.199591`
- 双方都成功: `1235`，双方都失败: `45`
- 消息不变占比: `9.63%`，平均每题少发消息: `2.7278`
- 平均每题少用 Token: `238.298`
- Token 分解: `prompt saving = 13.01%`，`completion saving = -2.60%`
- `Full` 优势样例: `GSM8K/45, GSM8K/98, GSM8K/151, GSM8K/193, GSM8K/299, GSM8K/434, GSM8K/442, GSM8K/530, GSM8K/539, GSM8K/556`
- `VIB` 优势样例: `GSM8K/89, GSM8K/172, GSM8K/241, GSM8K/304, GSM8K/340, GSM8K/409, GSM8K/493, GSM8K/527, GSM8K/852, GSM8K/973`
- 同次 trace 的 `Full only` 样例:
  - `GSM8K/45`: `full_msgs=6`，`vib_msgs=1`，`full_out=答案: 104`，`vib_out=答案: 74.4`
  - `GSM8K/98`: `full_msgs=6`，`vib_msgs=2`，`full_out=答案: 5`，`vib_out=答案: 2.5`
  - `GSM8K/151`: `full_msgs=6`，`vib_msgs=1`，`full_out=答案: 5`，`vib_out=答案: 5.5`
- 同次 trace 的 `VIB only` 样例:
  - `GSM8K/89`: `full_msgs=6`，`vib_msgs=2`，`full_out=答案: 18`，`vib_out=答案: 24`
  - `GSM8K/172`: `full_msgs=6`，`vib_msgs=3`，`full_out=答案: 43`，`vib_out=答案: 51`
  - `GSM8K/241`: `full_msgs=6`，`vib_msgs=4`，`full_out=答案: 3`，`vib_out=答案: 6`

## 对先前讨论的修正

- HumanEval 和 MBPP 的提升需要谨慎表述。如果净增益只有 1 题，就更适合写成“结果持平、略优于 Full”，而不是“显著提升”。
- GSM8K 的下降也不能只看 `-0.99pp`，还要看它对应的是多少条配对逆转样本，以及这些样本是否集中在某些推理模式上。
- “消息压缩率远高于 Token 节省率”这个判断方向通常是对的，但需要 `prompt/completion` 拆分数据支撑，不能只凭经验解释。
- MBPP 压缩率低不一定只是“代码任务信息密度高”，还可能与当前实验配置有关，例如 `max_steps=2`、三智能体上限接近固定 6 条消息、以及消息模板较短。

## 建议的下一步

- 优先分析 `full_only`/`vib_only` 的逐题模式，尤其是 GSM8K 的 `full_only` 逆转样本。
- 后续实验固定记录 `prompt/completion` 拆分，避免再对 Token 节省原因做不可证伪的推断。
- 若要引入更强的自适应机制，先验证当前客户端/模型是否真的支持 `logprobs`、低开销前缀探针和可复用 KV cache；否则相关讨论仍属于架构设想，而不是可直接落地的实验结论。
