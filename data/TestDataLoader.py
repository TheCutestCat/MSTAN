# this provide the test data for the models
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

def check(data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    return 0

class MultiSourceProcessDataset(Dataset):
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
        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(wind_x.values).to(torch.float32),torch.tensor(other_x.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

def TestDataLoader(batch_size,data_path,T0,tau,index_wind,index_other):
    # load the little data
    check(data_path)

    """
    [Btch tau M_wind + M_other]
    """
    df = pd.read_csv(data_path)

    MyDataset = MultiSourceProcessDataset(df,T0,tau,index_wind,index_other)
    loader = DataLoader(MyDataset, batch_size=batch_size, shuffle=True,drop_last = True)
    return loader

