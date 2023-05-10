import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KernelDensity


# 使用LSTM进行序列建模
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 训练LSTM模型
def train_lstm_model(model, train_data, train_labels, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()


# 使用核密度估计进行概率密度预测
def predict_density(lstm_model, kde, input_data):
    lstm_model.eval()
    with torch.no_grad():
        lstm_output = lstm_model(input_data).cpu().numpy()

    kde.fit(lstm_output)
    prediction = kde.sample()

    return prediction


# 创建一个简单的数据集
input_size = 1
sequence_length = 10
num_samples = 100

data = torch.randn(num_samples, sequence_length, input_size)
labels = torch.randn(num_samples, input_size)

# 创建并训练LSTM模型
hidden_size = 50
num_layers = 1
output_size = 1
learning_rate = 0.01
epochs = 100

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

train_lstm_model(lstm_model, data, labels, optimizer, criterion, epochs)

# 结合LSTM和核密度估计进行概率密度预测
bandwidth = 0.5
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)

# 生成一些新的输入数据进行预测
input_data = torch.randn(num_samples, sequence_length, input_size)

prediction = predict_density(lstm_model, kde, input_data)

print(prediction)