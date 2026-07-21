---
kind: external_dependency
name: NumPy 数值计算库
slug: numpy
category: external_dependency
category_hints:
    - vendor_identity
scope:
    - '**'
---

作为底层数值计算依赖，用于向量相似度计算（np.dot）、L2 范数归一化（np.linalg.norm）以及所有嵌入向量的数学运算。是 VIB 路由器中余弦相似度和 log-sum-exp 稳定计算的核心支撑。