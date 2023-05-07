# all the single models
from data.TestDataLoader import TestDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, tau, M_wind,M_other):
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

class LSTMencoder(nn.Module):
    """
    INPUT [Batch, T0, 2]
    OUTPUT [Batch, T0, d_model]
    使用里面的全连接层拉改变模型的维度
    """
    pass

class LSTMdecoder(nn.Module):
    """
    INPUT [Batch, tau, 2]
    OUTPUT [Batch, tau, d_model]
    使用里面的全连接层拉改变模型的维度
    """
    pass

class SelfAttention(nn.Module):
    """
    这个是在完成了 position_encoding 之后
    INPUT [Batch, To+tau,d_model]
    OUTPUT [Batch, tau, d_attention]
    只选取self-attention的后面部分
    使用全连接层改变模型的维度
    """
    pass

class MixtureDensity(nn.Module):
    """
    INPUT [batch, tau, d_attention]
    OUTPUT [batch, tau, m]
    最后输出的是 dense_1, dense_2 para
    可以选择单个输出，当然也是可以有更多地多步输出
    """
    pass
#这里面的残差模块我们先忽略掉

if __name__ =='__main__':
    data_path = '../data/test_small_data.csv'
    dataloader = TestDataLoader(data_path, 'MultiSourceProcess')
    T0 = 10
    tau = 5
    batch_size = 32
    M_wind = 3
    M_other = 6
    Mymodel = MultiSourceProcess(tau, M_wind,M_other)

    for batch, (en_x, wind_x, other_x, y) in enumerate(dataloader):
        Mymodel(wind_x,other_x)
        #终于通过一个测试了，好感动。。
