# set up a base dataset class for pytorch
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class DatasetMaker:
    # for time series data, default to last 30 days for test and next 30 days for validation
    def __init__(self, data_path, inited=False, val_days=30, test_days=30, seq_len=30):
        # assumes that the data path is the training data if all paths are given, else assumes it is the full data
        self.data_path = data_path
        self.val_days = val_days
        self.test_days = test_days
        self.seq_len = seq_len
        self.inited = inited
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data = None
        self.mean = None
        self.std = None
        self.x = None
        self.y = None

        # Load the data
        if inited:
            self.train_data = torch.load('train_data.pt')
            self.val_data = torch.load('val_data.pt')
            self.test_data = torch.load('test_data.pt')
        else:
            self.data = pd.read_csv(data_path)
            self.preprocess_data()

    def split_data(self):
        # Make all possible sequences
        total_sequences = len(self.x) - self.seq_len + 1
        if total_sequences <= 0 or total_sequences - self.val_days - self.test_days <= 0:
            raise ValueError("Not enough data to create sequences.")

        x_seq = []
        y_seq = []
        for i in range(total_sequences):
            x_seq.append(self.x[i:i + self.seq_len])
            y_seq.append(self.y[i + self.seq_len - 1])  # Target is open price of last day in sequence

        x_seq = torch.stack(x_seq)
        y_seq = torch.stack(y_seq)

        # Total number of sequences
        total = len(x_seq)

        # Indices for splitting
        test_start = total - self.test_days
        val_start = test_start - self.val_days

        # Slicing
        self.train_x = x_seq[:val_start]
        self.train_y = y_seq[:val_start]

        self.val_x = x_seq[val_start:test_start]
        self.val_y = y_seq[val_start:test_start]

        self.test_x = x_seq[test_start:]
        self.test_y = y_seq[test_start:]

        # Bundle
        self.train_data = (self.train_x, self.train_y)
        self.val_data = (self.val_x, self.val_y)
        self.test_data = (self.test_x, self.test_y)

    def preprocess_data(self):
        # convert 'date' column from datetime to 3 columns: year, month, day
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        self.data.drop(columns=['date'], inplace=True)

        # split the data into features and target variable
        self.x = self.data
        # the target variable is the open price, to create it, we need to shift the data by one day
        self.y = self.data['open'].shift(-1)

        # drop the last row of the data, as it will not have a target variable
        self.x = self.x[:-1]
        self.y = self.y[:-1]

        # normalize the features
        self.mean = self.x.mean(axis=0)
        self.std = self.x.std(axis=0)
        self.x = (self.x - self.mean) / self.std

        # convert to torch tensors
        self.x = torch.tensor(self.x.values, dtype=torch.float32)
        self.y = torch.tensor(self.y.values, dtype=torch.float32)

        # split the data into train, validation and test sets
        self.split_data()

        # save the data to torch file
        torch.save(self.train_data, 'train_data.pt')
        torch.save(self.val_data, 'val_data.pt')
        torch.save(self.test_data, 'test_data.pt')
    
    def get_train_data(self):
        return BaseDataset(self.train_data)
    
    def get_val_data(self):
        return BaseDataset(self.val_data)
    
    def get_test_data(self):
        return BaseDataset(self.test_data)
    
    def get_dataloaders(self, batch_size=32, shuffle=True):
        return {
            "train": DataLoader(self.get_train_data(), batch_size=batch_size, shuffle=shuffle),
            "val": DataLoader(self.get_val_data(), batch_size=batch_size, shuffle=False),
            "test": DataLoader(self.get_test_data(), batch_size=batch_size, shuffle=False),
        }
        
class BaseDataset(Dataset):
    def __init__(self, data):  # data is (x, y)
        self.x, self.y = data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]