import torch

# 创建一个维度为（32，5）的tensor
x = torch.randn((32, 5))

# 计算绝对值的平均数
abs_mean = torch.abs(x).mean()

# 打印结果
print("The mean of absolute values is:", abs_mean.item())