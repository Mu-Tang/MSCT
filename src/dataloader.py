import numpy as np
import torch
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MyDataset(Dataset):
    def __init__(self, path):
        # 读取csv文件
        df = pd.read_csv(path, header=0).values
        self.data = df[:, 2:-3]
        self.label = df[:, -3:]

        # 数据标准化处理
        self.data = MinMaxScaler().fit_transform(self.data)
        # 标签标准化处理
        mm_label = MinMaxScaler()
        scaler = mm_label.fit(self.label)
        self.label = scaler.transform(self.label)

        # noise_std = 0.01 * np.std(self.label)
        # noise = np.random.normal(0, noise_std, self.label.shape)
        # self.label = self.label + noise

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class MyDataloader():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset
        self.train_loader, self.test_loader = self.create_data_loader()

    def create_data_loader(self):
        data = self.dataset.data
        label = self.dataset.label

        test_index = self.config.test_index

        # 分离数据集 采用合并的方式进行处理
        train_data = np.concatenate((data[0:test_index], data[test_index + 370:]), axis=0)
        test_data = data[test_index: test_index + 370]

        train_label = np.concatenate((label[0:test_index], label[test_index + 370:]), axis=0)
        test_label = label[test_index: test_index + 370:]

        X_train_dataset = torch.tensor(train_data, dtype=torch.float32).to(self.config.device)
        X_test_dataset = torch.tensor(test_data, dtype=torch.float32).to(self.config.device)

        y_train_dataset = torch.tensor(train_label, dtype=torch.float32).to(self.config.device)
        y_test_dataset = torch.tensor(test_label, dtype=torch.float32).to(self.config.device)

        train_dataset = TensorDataset(X_train_dataset, y_train_dataset)
        test_dataset = TensorDataset(X_test_dataset, y_test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, test_loader
