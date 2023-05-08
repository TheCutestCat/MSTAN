# this provide the test data for the models
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
T0 = 10
tau = 5
batch_size = 32

index_wind = ['wind10', 'wind30', 'wind50']
index_other = ['angle10', 'angle30', 'angle50', 'temp', 'atmosphere',
               'humidity']

def check(data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")
    return 0

class MultiSourceProcessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)-(T0+tau)

    def __getitem__(self, index):
        en_x = self.data.loc[index:index+T0-1, ['wind10','angle10']]
        wind_x, other_x = self.data.loc[index+T0:index+T0+tau-1, index_wind],self.data.loc[index+T0:index+T0+tau-1, index_other]
        y = self.data.loc[index:index+T0+tau-1, 'power']
        return torch.tensor(en_x.values).to(torch.float32),torch.tensor(wind_x.values).to(torch.float32),torch.tensor(other_x.values).to(torch.float32), torch.tensor(y.values).to(torch.float32)

def TestDataLoader(data_path,ModuleName):
    # load the little data
    check(data_path)


    if ModuleName == 'MultiSourceProcess':
        """
        [Btch tau M_wind + M_other]
        """
        df = pd.read_csv(data_path)

        MyDataset = MultiSourceProcessDataset(df)
        loader = DataLoader(MyDataset, batch_size=batch_size, shuffle=True,drop_last = True)
        return loader


if __name__ == '__main__':
    data_path = 'test_small_data.csv'
    loader = TestDataLoader(data_path,'MultiSourceProcess')
    for en_x_batch, wind_x_batch, other_x_batch, y_batch in loader:
        print(f"en_x_batch.shape: {en_x_batch.shape},{type(en_x_batch.shape)}")
        print(f"wind_x_batch.shape: {wind_x_batch.shape}")
        print(f"other_x_batch.shape: {other_x_batch.shape}")
        print(f"y_batch.shape: {y_batch.shape}")
        break
    print("DONE")