# Step 1
# This file is to load data

import torch
from torch.utils import data
import pandas as pd
import numpy as np

class QAdataset(data.Dataset):

    def __init__(self, para_path, ques_path, label_path=None, test_mode=False):
        self.para = pd.read_csv(para_path, header=None).values.astype(np.float32)
        self.ques = pd.read_csv(ques_path, header=None).values.astype(np.float32)
        self.test_mode = test_mode
        if self.test_mode is False:
            self.label = pd.read_csv(label_path, header=None).values.astype(np.float32)\

    def __len__(self):
        return self.para.shape[0] # 有多少行

    def __getitem__(self, index): # 取batch_size
        # Load data and get label
        X = torch.from_numpy(self.para[index])
        h = torch.from_numpy(self.ques[index])
        y = self.label[index]
        return X, h, y
"""
if __name__ == "__main__":
    train_para_path = './train_para.csv'
    train_ques_path = './train_ques.csv'
    train_label_path = './train_label.csv'
    dataset = QAdataset(train_para_path, train_ques_path, train_label_path)
"""