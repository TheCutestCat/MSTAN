import torch
from scipy.stats import beta

# 定义 alpha 和 beta 参数
a = torch.rand(32, 48, 10)
b = torch.rand(32, 48, 10)

# 置信区间的端点
confidence_interval = 0.9
lower_bound_percentile = (1 - confidence_interval) / 2
upper_bound_percentile = 1 - lower_bound_percentile

# 计算置信区间
lower_bounds = beta.ppf(lower_bound_percentile, a, b)
upper_bounds = beta.ppf(upper_bound_percentile, a, b)

# 结果是形状为 (32, 48, 10) 的 NumPy 数组，其中每个元素表示相应 Beta 分布的 90% 置信区间的边界
print("90% 置信区间的下限：\n", lower_bounds)
print("90% 置信区间的上限：\n", upper_bounds)

