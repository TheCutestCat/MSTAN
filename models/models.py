# all the single models
import os

import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.distributions import Beta

from data.TestDataLoader import TestDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.config import *

class NonlinearDense(nn.Module):
    def __init__(self, Batch,tau,feature):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(Batch,tau,feature))
        self.bias = nn.Parameter(torch.randn(Batch,tau,feature))

    def forward(self, x):

        return torch.relu(torch.mul(x, self.weight) + self.bias)# Batch,tau,M_wind

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

        self.Dense_wind_softmax = NonlinearDense(self.batch, self.tau, self.M_wind)
        self.Dense_other_softmax = NonlinearDense(self.batch,self.tau,self.M_other)
        # 这个结果应该没问题的，还是需要继续去测试一下
    def forward(self, wind_x,other_x):
        X = torch.cat([wind_x,other_x],dim = 2)
        X_wind_softmax = torch.relu(self.linear_wind(X))
        X_other_softmax = torch.relu(self.linear_other(X))

        out_wind = self.Dense_wind(wind_x)
        out_wind_proba = F.softmax(X_wind_softmax,dim=2)
        out_wind = torch.sum(torch.mul(out_wind, out_wind_proba), dim=2, keepdim=True)
        out_other = self.Dense_other(other_x)
        out_other_proba = F.softmax(X_other_softmax, dim=2)
        out_other = torch.sum(torch.mul(out_other, out_other_proba), dim=2, keepdim=True)
        out = torch.cat([out_wind, out_other], dim=2)

        return out #batch tau 2

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, hidden = self.lstm(x)
        return output,hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        output, _ = self.lstm(x, hidden)
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

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        Q = self.query(x)  # shape: (batch_size, seq_len, output_dim)
        K = self.key(x)    # shape: (batch_size, seq_len, output_dim)
        V = self.value(x)  # shape: (batch_size, seq_len, output_dim)

        attention_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

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

def Loss(alpha, beta, pi, y):
    beta_dist = Beta(alpha, beta)# 添加一个小数，防止生成函数的时候因为0的存在导致出现了问题

    PdfValues = beta_dist.log_prob(y.unsqueeze(-1)).exp()  # (32, 5, 3)

    weighted_pdf_values = (pi * PdfValues).sum(dim=2)  # (32, 5)

    loss = -weighted_pdf_values.log().mean()

    return loss

def GetY_pre(alpha, beta, pi):
    # alpha * pi / (alpha + beta)
    y_pre = torch.div(torch.mul(alpha,pi), torch.add(alpha,beta)) #都是一对一的运算，(batch,tau,m)
    y_pre = y_pre.sum(dim = 2)
    return y_pre

class Ensemble(nn.Module):

    def __init__(self):
        super().__init__()

        self.tau = tau #这里有一个需要添加上去的
        self.dataloader = TestDataLoader(batch_size, data_path, T0, tau, index_wind, index_other)
        self.MultiProcess = MultiSourceProcess(batch_size, tau, M_wind, M_other)

        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder)
        self.self_attention = SelfAttention(input_dim, output_dim)
        self.mixtureDensity = MixtureDensity(d_attention, m)


    def forward(self,en_x, wind_x, other_x, y):
        Decoder_in = self.MultiProcess(wind_x, other_x)  # [batch,tau,2]
        Encoder_in = en_x  # [batch T0 2]

        output1, output2 = self.seq2seq(Encoder_in, Decoder_in)
        attention_in = torch.cat((output1, output2), dim=1)  # batch T0 + tau 64

        output = self.self_attention(attention_in)  # shape: (batch, T0 + tau, 16)
        output = output[:, -self.tau:, :]  # (batch, tau, 16)
        alpha, beta, pi = self.mixtureDensity(output)  # (batch, tau, 3) (batch, tau, 3) (batch, tau, 3)  the parameter for almost all of them
        Y = y[:, -self.tau:]  # (batch, tau,)#只是一个二维的，后面对其进行了扩充

        return alpha, beta, pi, Y

class trainer():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.myEnsemble = Ensemble().to(self.device)
        self.optimizer = optim.Adam(self.myEnsemble.parameters(), lr=learning_rate)
        self.dataloader = TestDataLoader(batch_size, data_path, T0, tau, index_wind, index_other)
        self.test_dataloader = None # TODO

        self.show_y = []
        self.show_y_pre = []
    def train(self,epoch = 1):
        self.myEnsemble.train()
        for i in range(epoch):
            loss_train = []
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader):
                # to device
                en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)
                loss = Loss(alpha, beta, pi, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(f'loss {loss}')

                loss_train.append(loss)
            loss_train = sum(loss_train)/len(loss_train)
            print(f'epoch {i+1}  {loss_train}')

    def test(self):
        self.myEnsemble.eval()
        with torch.no_grad():
            loss_test = []
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader):
                en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)
                #TODO use another test mode to analyse the y
                y_pre = GetY_pre(alpha, beta, pi) # TODO

                loss = torch.abs(y - y_pre).mean().item() #全部的平均数
                # print(f'batch {batch} loss {loss},has_nan_pre {has_nan_pre},has_nan_y {has_nan_y}')
                loss_test.append(loss)
            loss_test = sum(loss_test)/len(loss_test)
            print(f'loss is {loss_test}')
            return Loss

    def get_show_data(self,index):
        self.myEnsemble.eval()
        with torch.no_grad():
            self.dataloader = TestDataLoader(batch_size,data_path,T0,tau,index_wind,index_other,shuffle=False)
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader):
                if batch == index :
                    en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                    alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)

                    y_pre = GetY_pre(alpha, beta, pi)
                    show_y = y #[0,:]
                    show_y_pre = y_pre #[0,:]
                    show_y = show_y.cpu().detach().numpy()
                    show_y_pre = show_y_pre.cpu().detach().numpy()

                    self.show_y = show_y
                    self.show_y_pre = show_y_pre
                    break
    def show(self,index):
        show_y = self.show_y[index,:]
        show_y_pre = self.show_y_pre[index, :]

        plt.figure(figsize=(13, 6))
        plt.plot(show_y, linestyle='--', label='real')
        plt.plot(show_y_pre, linestyle='-', label='pre')
        plt.legend()
        plt.savefig('new.png')
        plt.show()
    def save(self,name = 'model'):
        path ='..\save'
        name = name + '.pt'
        path = os.path.join(path,name)
        torch.save(self.myEnsemble.state_dict(), path)
        print(f'model saved at {path}')

    def load(self,name = 'model'):
        path = '..\save'
        name = name+'.pt'
        path = os.path.join(path,name)
        self.myEnsemble.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')


if __name__ == '__main__':
    mytrainer = trainer()
    mytrainer.get_show_data()
    mytrainer.show(1)