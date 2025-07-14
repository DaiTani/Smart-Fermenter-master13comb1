from __future__ import print_function
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class FermentationData(Dataset):
    def __init__(self,
                 work_dir="./Data5",
                 train_mode=True,
                 y_var=["od_600"],
                 ws=20,
                 stride=5,
                 timestamp=None,
                 y_mean=None,
                 y_std=None):
        self.work_dir = work_dir
        self.train = train_mode
        self.ws = ws
        self.stride = stride
        self.timestamp = timestamp
        self.y_mean = y_mean
        self.y_std = y_std

        # 固定列名（必须与 Excel 第一行完全一致）
        self.x_var = [
            "dm_air", "m_ls_opt_do", "m_ph", "m_stirrer",
            "m_temp", "dm_o2", "dm_spump1", "dm_spump2",
            "dm_spump3", "dm_spump4", "induction"
        ]
        self.y_var = y_var
        self.all_cols = ["Timestamp"] + self.x_var + self.y_var

        # 训练 / 测试批次
        self.train_nums = [22]
        self.test_nums = [28]

        # 1. 加载数据
        self.X, self.Y = self._load_data()

        # 2. 归一化参数
        self._compute_norm()

        # 3. 预处理
        self.X = self._preprocess(self.X, self.ws, self.stride)
        self.Y = self._preprocess_labels(self.Y, self.ws, self.stride)

        # 4. 训练集洗牌
        if self.train:
            idx = np.random.permutation(len(self.X))
            self.X = self.X[idx]
            self.Y = self.Y[idx]

    def _load_data(self):
        nums = self.train_nums if self.train else self.test_nums
        X, Y = [], []
        for n in nums:
            folder = "train" if self.train else "test"
            file_path = os.path.join(self.work_dir, folder, "data_clean.xlsx")
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
            df = df[self.all_cols]
            X.append(df[self.x_var].values.astype(np.float32))
            Y.append(df[self.y_var].values.astype(np.float32))
        return np.array(X, dtype=object), np.array(Y, dtype=object)

    def _compute_norm(self):
        flat_X = np.concatenate([np.asarray(x, dtype=np.float32) for x in self.X], axis=0)
        self.x_mean = flat_X.mean(axis=0)
        self.x_std = flat_X.std(axis=0) + 1e-8

        flat_Y = np.concatenate([np.asarray(y, dtype=np.float32) for y in self.Y], axis=0)
        self.y_mean = float(flat_Y.mean())
        self.y_std = float(flat_Y.std()) + 1e-8

    def _preprocess(self, X, ws, stride):
        out = []
        for sample in X:
            sample = (sample - self.x_mean) / self.x_std
            seqs = [sample[i:i + ws] for i in range(0, len(sample) - ws + 1, stride)]
            out.extend(seqs)  # 使用 extend 而不是 append
        return np.array(out, dtype=np.float32)  # 转换为标准 NumPy 数组

    def _preprocess_labels(self, Y, ws, stride):
        out = []
        for y in Y:
            y = (y - self.y_mean) / self.y_std
            seqs = [y[i + ws - 1] for i in range(0, len(y) - ws + 1, stride)]
            out.extend(seqs)  # 使用 extend 而不是 append
        return np.array(out, dtype=np.float32).reshape(-1, 1)  # 转换为标准 NumPy 数组

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.Y[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def get_num_features(self):
        return self.X.shape[-1]