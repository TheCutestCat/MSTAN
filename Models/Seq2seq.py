import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from data.TestDataLoader import loader_seq2seq_value,loader_show
from config import *
from Models.models import Encoder, Decoder, Seq2Seq, EarlyStopping

from utils.tools import set_seed,Loss_value

# ['wind10', 'temp', 'atmosphere', 'humidity', 'power', 'en_month','en_hour', 'en_cos_angle']

class Seq2seq(nn.Module):
    def __init__(self):
        super().__init__()
        # 注意处理好参数的问题
        self.T0 = T0
        self.tau =tau
        self.batch = batch_size
        self.hidden_size = hidden_size
        self.input_size = 7 #就是输入dim = -1的size 这里我们采用 wind10 angle10来计算
        self.output_size = 7 #这个好像也是同样的输入
        # model
        self.encoder = Encoder(self.input_size,self.hidden_size)
        self.decoder = Decoder(self.hidden_size,self.output_size) #长度在这里是无关的
        self.seq2seq = Seq2Seq(self.encoder, self.decoder)
        self.linear = nn.Linear(self.hidden_size,1) #最终的输出为1

    def forward(self,Encoder_in,Decoder_in,y):

        _, y_pre = self.seq2seq(Encoder_in, Decoder_in)
        y_pre = self.linear(y_pre).squeeze()
        return y,y_pre

class trainer_seq2seq():
    def __init__(self,learning_rate = 0.001):
        set_seed(seed = 42)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.mymodel = Seq2seq().to(self.device)
        self.optimizer = optim.Adam(self.mymodel.parameters(), lr=learning_rate)
        self.dataloader_train, self.dataloader_test = loader_seq2seq_value(batch_size, data_path, T0, tau)
        self.dataloader_show = loader_show(data_path,T0,tau,index_begin = 100,index_end = 200)
        self.early_stopping = EarlyStopping()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

    def train(self,epoch = 1,early_stop_patience = 4):

        for i in range(epoch):
            self.mymodel.train()
            loss_train = []
            for batch, (en_x, en_x_pre, y) in enumerate(self.dataloader_train):
                # to device
                en_x, en_x_pre, y = en_x.to(self.device), en_x_pre.to(self.device), y.to(self.device)
                Y,Y_pre = self.mymodel(en_x, en_x_pre, y)
                loss = Loss_value(Y_pre,Y)
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
        self.mymodel.eval()
        with torch.no_grad():
            loss_test = []
            for batch, (en_x, en_x_pre, y) in enumerate(self.dataloader_test):
                en_x, en_x_pre, y = en_x.to(self.device), en_x_pre.to(self.device), y.to(self.device)
                Y_pre, Y = self.mymodel(en_x, en_x_pre, y)
                loss = Loss_value(Y_pre, Y)
                loss_test.append(loss)
            loss_test = sum(loss_test) / len(loss_test)
            return loss_test

    def show(self,show =True,index_begin= 100, index_end = 200):
        dataloader_show = loader_show(data_path, T0, tau, index_begin, index_end)
        Y_target = []
        Y_forcast = []
        for i, (en_x, en_x_pre, y) in enumerate(dataloader_show):
            en_x, en_x_pre, y = en_x.to(self.device), en_x_pre.to(self.device), y.to(self.device)
            Y, Y_pre = self.mymodel(en_x, en_x_pre, y)
            Y,Y_pre = Y.cpu().squeeze().tolist(),Y_pre.cpu().squeeze().tolist()
            Y_target += Y
            Y_forcast += Y_pre
        if(show):
            plt.figure(figsize=(13, 6))
            plt.plot(Y_target, linestyle='-',color='blue', label='real')
            plt.plot(Y_forcast, linestyle='--',color='red', label='pre')
            plt.legend()
            plt.savefig('save/picture')
            plt.show()
        return Y_target,Y_forcast

    def save(self,name = 'model'):
        path ='save'
        name = name + '.pt'
        path = os.path.join(path,name)
        torch.save(self.mymodel.state_dict(), path)
        print(f'model saved at {path}')

    def load(self,name = 'model'):
        path = 'save'
        name = name+'.pt'
        path = os.path.join(path,name)
        self.mymodel.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')


