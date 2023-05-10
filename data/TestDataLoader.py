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


def loader_proba(batch_size,data_path,T0,tau,index_wind,index_other):
    check(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = df.loc[:0.8*len(df)],df.loc[0.8*len(df):]

    MyDataset_train = MSTAN_proba(train_df,T0,tau,index_wind,index_other)
    MyDataset_test = MSTAN_proba(train_df,T0,tau,index_wind,index_other)

    loader_train = DataLoader(MyDataset_train, batch_size=batch_size, shuffle=True,drop_last = True)
    loader_test = DataLoader(MyDataset_test, batch_size=batch_size, shuffle=False,drop_last = True)

    return loader_test,loader_train

def loader_value(batch_size,data_path,T0,tau,index_wind,index_other):
    check(data_path)
    df = pd.read_csv(data_path)
    train_df, test_df = df.loc[:0.8*len(df)],df.loc[0.8*len(df):]

    MyDataset_train = MSTAN_value(train_df,T0,tau,index_wind,index_other)
    MyDataset_test = MSTAN_value(train_df,T0,tau,index_wind,index_other)

    loader_train = DataLoader(MyDataset_train, batch_size=batch_size, shuffle=True,drop_last = True)
    loader_test = DataLoader(MyDataset_test, batch_size=batch_size, shuffle=False,drop_last = True)

    return loader_test,loader_train





