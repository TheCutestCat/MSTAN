import torch
import torch.nn as nn

# 定义输入张量
x = torch.randn(2, 4, 8).to('cuda')

# 定义 Layer Normalization 的维度
normalized_shape = 8

x = nn.LayerNorm(x.size()[1:]).to('cuda')(x)

print(x.shape)  # 输出：torch.Size([2, 4, 8])