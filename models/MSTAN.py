import numpy as np
import torch
from scipy.stats import beta

# 定义一个复杂的运算
def func(last_two_cols):
    a, b = last_two_cols[..., 0], last_two_cols[..., 1]
    result = pdf = beta.pdf(0.5, a, b)

    return result

# 生成一个维度为（32，5，3）的矩阵
matrix = np.random.rand(32, 5, 3)
matrix = torch.tensor(matrix)
# 提取最后两个维度
last_two_dims = matrix[:, :, 1:]

# 对最后两个维度进行运算
result = func(last_two_dims)

# 输出结果
print(result.shape)  # (32, 5)