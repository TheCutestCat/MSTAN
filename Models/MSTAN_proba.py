import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

from config import *
from data.TestDataLoader import loader_proba
from Models.models import MultiSourceProcess, Encoder, Decoder, Seq2Seq, SelfAttention, MixtureDensity, \
    ResidualNet, Norm, PositionalEncoding, EarlyStopping
from utils.tools import Loss_proba, GetY_pre, set_seed


class MSTAN_proba(nn.Module):

    def __init__(self):
        super().__init__()

        self.tau = tau
        #对模型进行init的初始化构造
        self.MultiProcess = MultiSourceProcess(batch_size, tau, M_wind, M_other)
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.seq2seq = Seq2Seq(self.encoder, self.decoder)
        self.self_attention = SelfAttention(input_dim, output_dim)
        self.mixtureDensity = MixtureDensity(input_mix, m)
        self.res1 = ResidualNet(input_size = 2, output_size = input_dim)
        self.res2 = ResidualNet(input_size=input_dim, output_size=input_mix)
        self.norm1 = Norm(input_dim)
        self.norm2 = Norm(input_mix)
        self.PE = PositionalEncoding(input_dim)

    def forward(self,en_x, wind_x, other_x, y):
        Decoder_in = self.MultiProcess(wind_x, other_x)  # [batch,tau,2]
        Encoder_in = en_x  # [batch T0 2]

        output1, output2 = self.seq2seq(Encoder_in, Decoder_in) #这里面添加了GLU模块 batch T0 input_dim batch tau input_dim 32 48 32
        output1 = torch.add(output1,self.res1(Encoder_in)) # batch T0 input_dim  32 48 32
        output2 = torch.add(output2,self.res1(Decoder_in)) #ResidualNet
        output1= self.norm1(output1)
        output2 = self.norm1(output2)
        output = torch.cat((output1, output2), dim=1)  # batch T0 + tau input_dim 32 96 32

        output = self.PE(output)
        output = self.self_attention(output)  # shape: (batch, T0 + tau, input_mix)  32 96 8
        output = output[:, -self.tau:, :]  # (batch, tau, input_mix)32 48 8

        output2 = self.res2(output2)
        output = torch.add(output,output2)
        output = self.norm2(output)# (batch, tau, input_mix)32 48 8
        alpha, beta, pi = self.mixtureDensity(output)  # (batch, tau, 3) (batch, tau, 3) (batch, tau, 3)  the parameter for almost all of them
        Y = y[:, -self.tau:]  # (batch, tau,)#只是一个二维的，后面对其进行了扩充

        return alpha, beta, pi, Y

class trainer_MSTAN_proba():
    def __init__(self,learning_rate = 0.001):
        set_seed(seed = 42)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.myEnsemble = MSTAN_proba().to(self.device)
        self.optimizer = optim.Adam(self.myEnsemble.parameters(), lr=learning_rate)
        self.dataloader_train,self.dataloader_test  = loader_proba(batch_size, data_path, T0, tau, index_wind, index_other)
        self.early_stopping = EarlyStopping()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.show_y = []
        self.show_y_pre = []

    def train(self,epoch = 1,early_stop_patience = 4):

        for i in range(epoch):
            self.myEnsemble.train()
            loss_train = []
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader_train):
                # to device
                en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)
                loss = Loss_proba(alpha, beta, pi, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_train.append(loss)
            self.lr_scheduler.step()
            loss_train = sum(loss_train)/len(loss_train)
            loss_test = self.test()

            self.early_stopping(loss_test,patience = early_stop_patience)
            if self.early_stopping.stop_training:
                print("Early Stopping!")
                break
            print(f'epoch {i+1:>3}  loss_train : {loss_train:.3f}, loss_test : {loss_test:.4f}, learning_rate :{self.lr_scheduler.get_last_lr()[0]:.6f}')

    def test(self):
        self.myEnsemble.eval()
        with torch.no_grad():
            loss_test = []
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader_test):
                en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)
                loss = Loss_proba(alpha, beta, pi, y)
                loss_test.append(loss)
            loss_test = sum(loss_test)/len(loss_test)
            return loss_test

    def get_show_data(self,index,confidence):
        self.myEnsemble.eval()
        with torch.no_grad():
            for batch, (en_x, wind_x, other_x, y) in enumerate(self.dataloader_test):
                if batch == index :
                    en_x, wind_x, other_x, y = en_x.to(self.device), wind_x.to(self.device), other_x.to(self.device), y.to(self.device)
                    alpha, beta, pi, y = self.myEnsemble(en_x, wind_x, other_x, y)
                    alpha, beta, pi, y = alpha.cpu(), beta.cpu(), pi.cpu(), y.cpu()
                    lower_bounds,upper_bounds = GetY_pre(alpha, beta, pi,confidence=confidence)

                    show_y = y #[0,:]
                    show_y_pre_lower = lower_bounds
                    show_y_pre_upper = upper_bounds
                    show_y = show_y.cpu().detach().numpy()

                    self.show_y = show_y
                    self.show_y_pre_lower = show_y_pre_lower
                    self.show_y_pre_upper = show_y_pre_upper
                    break
    def show(self,index):
        show_y = self.show_y[index,:]
        show_y_pre_lower = self.show_y_pre_lower[index, :]
        show_y_pre_upper = self.show_y_pre_upper[index, :]

        plt.figure(figsize=(13, 6))
        plt.plot(show_y, linestyle='-', label='real')
        plt.plot(show_y_pre_lower, linestyle='--',color='yellow', label='pre')
        plt.plot(show_y_pre_upper, linestyle='--',color='yellow', label='pre')
        plt.ylim(0,1)
        plt.legend()
        plt.savefig('new.png')
        plt.show()
    def save(self,name = 'model'):
        path ='save'
        name = name + '.pt'
        path = os.path.join(path,name)
        torch.save(self.myEnsemble.state_dict(), path)
        print(f'model saved at {path}')

    def load(self,name = 'model'):
        path = 'save'
        name = name+'.pt'
        path = os.path.join(path,name)
        self.myEnsemble.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')