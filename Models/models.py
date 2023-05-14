# all the single Models
import math
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonlinearDense(nn.Module):
    def __init__(self, Batch,tau,feature):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(tau, feature))
        init.normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros(tau, feature))
        self.Batch = Batch
        self.tau = tau
        self.feature = feature

        #需要是同样的参数的
    def forward(self, x):
        weight = self.weight.expand(self.Batch, self.tau, self.feature)
        bias = self.bias.expand(self.Batch, self.tau, self.feature)
        # 最终会只去更新作为parameter的参数

        return torch.relu(torch.mul(x, weight) + bias)# Batch,tau,M_wind

class MultiSourceProcess(nn.Module):
    """
    input [Batch tau M_wind + M_other]
    output [Batch tau 2] 相当于每一个来源的信号都输出一个风速 与 一个相关系数（风向）
    原来是一个就能输出这么多的结果呀。。最终的计算结果还是非常快的，
    我们只去做一个d_model 相当于只输出一个d_model还有一个别的结构
    """
    def __init__(self,batch_size , tau, M_wind,M_other):
        super(MultiSourceProcess, self).__init__()
        self.batch = batch_size
        self.tau = tau
        self.M_wind = M_wind
        self.M_other = M_other

        self.linear_wind = nn.Linear(self.M_wind + self.M_other, self.M_wind)
        self.linear_other = nn.Linear(self.M_wind + self.M_other, self.M_other)

        self.Dense_wind = NonlinearDense(self.batch,self.tau,self.M_wind)
        self.Dense_other = NonlinearDense(self.batch,self.tau,self.M_other)

        init.normal_(self.linear_wind.weight)
        init.zeros_(self.linear_wind.bias)
        init.normal_(self.linear_other.weight)
        init.zeros_(self.linear_other.bias)

        # 这个结果应该没问题的，还是需要继续去测试一下
    def forward(self, wind_x,other_x):
        X = torch.cat([wind_x,other_x],dim = 2)
        X_wind_softmax = self.linear_wind(X)
        X_other_softmax = self.linear_other(X)

        out_wind = self.Dense_wind(wind_x)
        out_wind_proba = F.softmax(X_wind_softmax,dim=2)
        out_wind = torch.sum(torch.mul(out_wind, out_wind_proba), dim=2, keepdim=True)
        out_other = self.Dense_other(other_x)
        out_other_proba = F.softmax(X_other_softmax, dim=2)
        out_other = torch.sum(torch.mul(out_other, out_other_proba), dim=2, keepdim=True)
        out = torch.cat([out_wind, out_other], dim=2)

        return out #batch tau 2

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.glu(x, dim=self.dim)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.glu = GLU(2)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, hidden = self.lstm(x)
        # output = self.glu(output)
        return output,hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.glu = GLU(2)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        output, _ = self.lstm(x, hidden)
        # output = self.glu(output)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        output1, hidden = self.encoder(source)
        output2 = self.decoder(target, hidden)
        return output1,output2

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.scale_factor = torch.sqrt(torch.tensor(output_dim, dtype=torch.float32))
        self.glu = GLU(2)
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        Q = self.query(x)  # shape: (batch_size, seq_len, output_dim)
        K = self.key(x)    # shape: (batch_size, seq_len, output_dim)
        V = self.value(x)  # shape: (batch_size, seq_len, output_dim)

        attention_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = self.glu(attention_output)
        return attention_output

class Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.linear(x)
        output = self.relu(output)
        return output

class MixtureDensity(nn.Module):
    """
    INPUT [batch, tau, d_attention]
    OUTPUT [batch, tau, m]
    最后输出的是 dense_1, dense_2 para
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.alpha = Dense(input_size, output_size)
        self.beta = Dense(input_size,output_size)
        self.pi = Dense(input_size,output_size)
    def forward(self,x):
        # todo 防止再出现nan
        alpha = self.alpha(x) + 1
        beta = self.beta(x) + 1

        pi = self.pi(x)
        pi = torch.softmax(pi,dim = 2)
        return alpha, beta, pi

class ResidualNet(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear(x)
        x = self.relu(x) #may be not very necessary
        return x

class Norm(nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.norm = nn.LayerNorm(shape)
    def forward(self,x):
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        if(d_model%2 !=0):
            raise ValueError("d_model is not pair, change the d_model to pair")
        self.dropout = nn.Dropout(p=0.1)

        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # 添加位置编码到输入张量中
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EarlyStopping:
    def __init__(self, patience=4, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, val_loss,patience):
        self.patience = patience
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True
        else:
            self.best_score = val_loss
            self.counter = 0

class mixtureDensity_value(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.linear = nn.Linear(input_size,1)
    def forward(self,x):
        x = self.linear(x)
        return x

