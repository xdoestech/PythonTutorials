''''
use custom dataset
extend dataset class (torch.utils.data.Dataset	)
allows data set to be used with torch.utils.data.DataLoader	


All subclasses of the Dataset class must override
__len__: provides the size of the dataset,
__getitem__: supporting integer indexing in range from 0 to len(self) exclusive.
'''

import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd


class OHLC(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    def __getitem__(self, index):
        r=self.data.iloc[index]
        label = torch.tensor(r.is_up_day, dtype=torch.long)
        sample = self.normalize(torch.tensor([r.open, r.high, r.low, r.close]))
    def __len__(self):
        return len(self.data)