# this provide the test data for the Models
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def check(data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    return 0

class MSTAN_proba(Dataset):
    def __init__(self, data,T0,tau,index_wind,index_other):
        self.data = data
        self.T0 = T0
        self.tau = tau
        self.index_wind = index_wind
        self.index_other = index_other

    def __len__(self):
        return len(self.data)-(self.T0+self.tau)

    def __getitem__(self, index):
        en_x = self.data.loc[index:index+self.T0-1, ['wind10','angle10']]
        wind_x, other_x = self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_wind],self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_other]
        y = self.data.loc[index:index+self.T0+self.tau-1, 'power']
        y = ((y + 0.1) / 200) #进行一个大致的处理
        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(wind_x.values).to(torch.float32),torch.tensor(other_x.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

class MSTAN_value(Dataset):
    def __init__(self, data,T0,tau,index_wind,index_other):
        self.data = data
        self.T0 = T0
        self.tau = tau
        self.index_wind = index_wind
        self.index_other = index_other

    def __len__(self):
        return len(self.data)-(self.T0+self.tau)

    def __getitem__(self, index):
        en_x = self.data.loc[index:index+self.T0-1, ['wind10','angle10']]
        wind_x, other_x = self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_wind],self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_other]
        y = self.data.loc[index:index+self.T0+self.tau-1, 'power']
        # y = ((y + 0.1) / 200)
        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(wind_x.values).to(torch.float32),torch.tensor(other_x.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

class Seq2seq_value(Dataset):
    def __init__(self, data,T0,tau):
        self.data = data
        self.T0 = T0
        self.tau = tau
        self.index_seq2seq = ['wind10', 'temp', 'atmosphere', 'humidity', 'en_month','en_hour', 'en_cos_angle']

    def __len__(self):
        return len(self.data)-(self.T0+self.tau)

    def __getitem__(self, index):
        en_x = self.data.loc[index:index+self.T0-1, self.index_seq2seq]
        en_x_pre = self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_seq2seq]
        y = self.data.loc[index+self.T0:index+self.T0+self.tau-1, 'power'] #tau

        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(en_x_pre.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

#this might not be very sueful
class Seq2seq_value_moredata(Dataset):
    def __init__(self, data,T0,tau):
        self.data = data
        self.T0 = T0
        self.tau = tau
        # ['wind10', 'temp', 'atmosphere', 'humidity', 'power', 'en_month','en_hour', 'en_cos_angle']
        self.index_seq2seq = ['wind10', 'temp', 'atmosphere', 'humidity', 'en_month','en_hour', 'en_cos_angle']

    def __len__(self):
        return len(self.data)-(self.T0+self.tau)

    def __getitem__(self, index):
        en_x = self.data.loc[index:index+self.T0-1, self.index_seq2seq]
        en_x_pre = self.data.loc[index+self.T0:index+self.T0+self.tau-1, self.index_seq2seq]
        y = self.data.loc[index+self.T0:index+self.T0+self.tau-1, 'power'] #tau

        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(en_x_pre.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

class DatasetShow(Dataset):
    def __init__(self,data,T0,tau,index_begin,index_end):
        self.data = data
        self.T0 = T0
        self.tau = tau
        self.index_begin = index_begin
        self.index_end = index_end
        if(self.index_begin <= self.T0):
            raise ValueError('idnex_begin should lager than T0')
        if(self.index_end > len(self.data) + self.tau or self.index_end < self.index_begin + self.tau):
            raise ValueError('index_end out of bound')
    def __len__(self):
        return int(1 + (self.index_end - self.index_begin)/self.tau) #控制len来控制最后的长度
    def __getitem__(self, index):
        # index_en_x = [index * self.tau, index * self.tau+self.T0-1]
        # index_en_x_pre = [index * self.tau+self.T0, index * self.tau+self.T0+self.tau-1]
        column_x = ['wind10', 'temp', 'atmosphere', 'humidity', 'en_month','en_hour', 'en_cos_angle']
        en_x = self.data.loc[self.index_begin+index * self.tau: self.index_begin+index * self.tau+self.T0-1,column_x] #
        en_x_pre = self.data.loc[self.index_begin+index * self.tau++self.T0: self.index_begin+index * self.tau + self.T0+self.tau - 1,column_x]
        y = self.data.loc[self.index_begin+index * self.tau+self.T0 : self.index_begin+index * self.tau + self.T0+self.tau - 1, 'power']
        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(en_x_pre.values).to(torch.float32),torch.tensor(y.values).to(torch.float32)

def loader_proba(batch_size,data_path,T0,tau,index_wind,index_other):
    check(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = df.loc[:0.8*len(df)],df.loc[0.8*len(df):]

    MyDataset_train = MSTAN_proba(train_df,T0,tau,index_wind,index_other)
    MyDataset_test = MSTAN_proba(train_df,T0,tau,index_wind,index_other)

    loader_train = DataLoader(MyDataset_train, batch_size=batch_size, shuffle=True,drop_last = True)
    loader_test = DataLoader(MyDataset_test, batch_size=batch_size, shuffle=False,drop_last = True)

    return loader_train,loader_test

def loader_value(batch_size,data_path,T0,tau,index_wind,index_other):
    check(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = df.loc[:0.8*len(df)],df.loc[0.8*len(df):]

    MyDataset_train = MSTAN_value(train_df,T0,tau,index_wind,index_other)
    MyDataset_test = MSTAN_value(train_df,T0,tau,index_wind,index_other)

    loader_train = DataLoader(MyDataset_train, batch_size=batch_size, shuffle=True,drop_last = True)
    loader_test = DataLoader(MyDataset_test, batch_size=batch_size, shuffle=False,drop_last = True)

    return loader_train,loader_test

def loader_seq2seq_value(batch_size,data_path,T0,tau):
    check(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = df.loc[:0.8*len(df)],df.loc[0.8*len(df):]

    MyDataset_train = Seq2seq_value(train_df,T0,tau)
    MyDataset_test = Seq2seq_value(train_df,T0,tau)

    loader_train = DataLoader(MyDataset_train, batch_size=batch_size, shuffle=True,drop_last = True)
    loader_test = DataLoader(MyDataset_test, batch_size=batch_size, shuffle=False,drop_last = True)

    return loader_train,loader_test

def loader_show(data_path,T0,tau,index_begin,index_end,batch_size = 1):
    check(data_path)
    df = pd.read_csv(data_path)
    Dataset = DatasetShow(df,T0,tau,index_begin,index_end)
    loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader

def show(data_path,T0,index_begin,index_end):
    check(data_path)
    df = pd.read_csv(data_path)
    df = df.loc[index_begin+T0:index_end+T0,'power']
    return df.squeeze().tolist()

