import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

# 假设你有以下输入，这里我使用随机值来代替
alpha = torch.rand(32, 5, 3)
beta = torch.rand(32, 5, 3)
pi = torch.rand(32, 5, 3)

# Softmax 应用在第三个维度
pi = F.softmax(pi, dim=2)

# 假设你有一个 ground truth 概率值向量 y，维度为 (32, 5)
y = torch.rand(32, 5)
def Loss(alpha,beta,pi,y):
    beta_dist = Beta(alpha, beta)

    pdf_values = beta_dist.log_prob(y.unsqueeze(-1)).exp() # (32, 5, 3)

    weighted_pdf_values = (pi * pdf_values).sum(dim=2) #(32, 5)

    loss = -weighted_pdf_values.log().mean()

    return loss

loss = Loss(alpha,beta,pi,y)