import torch
import torch.nn as nn


# 定义Self-Attention网络
class SelfAttentionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(SelfAttentionNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.embedding(x)
        out = out.permute(1, 0, 2)
        out, _ = self.attention(out, out, out)
        out = out.permute(1, 0, 2)
        out = self.relu(out)
        out = self.fc(out)
        return out.squeeze()


# 定义测试Self-Attention网络
input_size = 10
hidden_size = 20
num_heads = 2
num_layers = 2
seq_len = 5
batch_size = 2

model = SelfAttentionNet(input_size, hidden_size, num_heads, num_layers)
x = torch.randn(seq_len, batch_size, input_size)
out = model(x)
print(out.shape)  # (2,)