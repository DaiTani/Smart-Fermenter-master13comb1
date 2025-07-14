from __future__ import print_function
import os
from os.path import join
import numpy as np
import sys
import torch
from torch.utils.data import Dataset
import glob
import pdb
import pandas as pd
import math

# ========== 工具函数实现 ==========
def load_data(work_dir, fermentation_number, data_file="data.xlsx", x_cols=None, y_cols=None):
    """加载单个发酵数据"""
    data_path = os.path.join(work_dir, str(fermentation_number), data_file)
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"加载数据文件失败: {data_path}")
        print(f"错误信息: {str(e)}")
        return (None, None)
    
    # 打印文件中的实际列名用于调试
    print(f"数据文件 {data_path} 中的列名: {df.columns.tolist()}")
    
    # 动态确定特征列：排除Timestamp和od_600
    available_columns = df.columns.tolist()
    
    # 如果未指定x_cols，则使用除Timestamp和od_600外的所有列
    if x_cols is None:
        x_cols = [col for col in available_columns if col not in ["Timestamp", "od_600"]]
    
    # 检查缺失的列
    missing_cols = [col for col in x_cols if col not in available_columns]
    if missing_cols:
        print(f"警告: 数据文件中缺少以下列: {missing_cols}")
        # 只保留数据文件中存在的列
        x_cols = [col for col in x_cols if col in available_columns]
    
    # 提取特征和标签
    X = df[x_cols].values if x_cols else None
    
    # 确保y_cols存在
    if y_cols is None:
        y_cols = ["od_600"]
    else:
        y_cols = [col for col in y_cols if col in available_columns]
        if not y_cols:
            print("警告: 未找到任何标签列，使用默认的od_600")
            y_cols = ["od_600"]
    
    Y = df[y_cols].values if y_cols else None
    
    return (X, Y)

def get_norm_param(X, x_cols=None, y_cols=None):
    """计算归一化参数"""
    # 计算均值和标准差
    if X is None or len(X) == 0:
        return 0, 1
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    
    # 避免除零错误
    std[std == 0] = 1.0
    
    return mean, std

def cumulative2snapshot(cumulative_data):
    """将累积数据转换为瞬时数据"""
    if cumulative_data is None or len(cumulative_data) == 0:
        return np.array([])
    
    snapshot = np.zeros_like(cumulative_data)
    if len(cumulative_data) > 0:
        snapshot[0] = cumulative_data[0]
        for i in range(1, len(cumulative_data)):
            snapshot[i] = cumulative_data[i] - cumulative_data[i-1]
    return snapshot

def normalise(X, mean, std, mode="z-score", binary_var=None):
    """归一化数据"""
    if X is None or len(X) == 0:
        return X
    
    if binary_var is None:
        binary_var = []
    
    if mode == "z-score":
        # 创建归一化数据的副本
        X_norm = np.copy(X)
        
        # 处理每一列
        for col_idx in range(X.shape[1]):
            if col_idx < len(mean) and col_idx < len(std) and col_idx not in binary_var:
                # 避免除零错误
                if std[col_idx] == 0:
                    X_norm[:, col_idx] = 0
                else:
                    X_norm[:, col_idx] = (X[:, col_idx] - mean[col_idx]) / std[col_idx]
        return X_norm
    else:
        return X  # 其他归一化模式未实现

def data2sequences(X, ws=20, stride=10):
    """将数据转换为序列"""
    if X is None or len(X) == 0 or len(X) < ws:
        return np.array([])
    
    sequences = []
    for i in range(0, len(X) - ws + 1, stride):
        sequences.append(X[i:i+ws])
    return np.array(sequences)

def polynomial_interpolation(data):
    """多项式插值 - 简单实现"""
    if data is None or len(data) == 0:
        return np.array([])
    return np.interp(np.arange(len(data)), np.arange(len(data)), data)

def linear_local_interpolation(data):
    """线性局部插值 - 简单实现"""
    if data is None or len(data) == 0:
        return np.array([])
    return np.interp(np.arange(len(data)), np.arange(len(data)), data)

def mix_interpolation(data):
    """混合插值 - 简单实现"""
    if data is None or len(data) == 0:
        return np.array([])
    return np.interp(np.arange(len(data)), np.arange(len(data)), data)

# ========== 数据集类 ==========
class FermentationData(Dataset):
    def __init__(self, work_dir="./Data", train_mode=True, y_var=["od_600"], ws=20, stride=5, timestamp=None, 
                 y_mean=None, y_std=None, skip_y_cumulative=False):
        self.work_dir = work_dir
        self.train = train_mode
        self.timestamp = timestamp
        self.y_mean = y_mean
        self.y_std = y_std
        self.skip_y_cumulative = skip_y_cumulative
        print(f"y_mean: {y_mean}, y_std: {y_std}, skip_y_cumulative: {skip_y_cumulative}")
        self.saved_windows = None
        self.last_row_index = None
        self.x_var = None  # 将在加载数据后确定
        
        print("Loading dataset...")
        
        # lists of number of fermentations for training and testing
        self.train_fermentations = [8, 11, 12, 16, 22, 23, 24, 25, 26, 27]
        self.test_fermentations = [28]

        # variables with cumulative values
        self.cumulative_var = [
            "dm_o2",
            "dm_air",
            "dm_spump1",
            "dm_spump2",
            "dm_spump3",
            "dm_spump4",
        ]
        # variables with binary values
        self.binary_var = ["induction"]

        # Using fermentation 16 for computing normalisation parameters
        self.fermentation_norm_number = 22
        self.X_norm, self.Y_norm = self.load_data(
            fermentation_number=self.fermentation_norm_number
        )
        
        # 确保数据加载成功
        if self.X_norm is None or len(self.X_norm) == 0:
            raise ValueError(f"无法加载归一化数据 (发酵 {self.fermentation_norm_number})")
        
        self.X_norm = self.X_norm[0]
        self.X_norm = self.cumulative2snapshot(self.X_norm)
        
        # 关键修复：跳过Y的累积处理
        self.Y_norm = self.Y_norm[0]
        if self.Y_norm is not None and len(self.Y_norm) > 0:
            if self.Y_norm.ndim == 1:
                self.Y_norm = self.Y_norm.reshape(-1, 1)  # 确保是二维数组
                
            if not self.skip_y_cumulative:
                self.Y_norm = self.cumulative2snapshot(self.Y_norm)
            else:
                print("跳过Y的累积处理")
        else:
            self.Y_norm = np.zeros((len(self.X_norm), 1))  # 创建默认Y数据
            print("警告: 没有Y数据，创建默认值")

        # Loading data
        self.X, self.Y = self.load_data()
        
        # 确保数据加载成功
        if self.X is None or len(self.X) == 0:
            raise ValueError("无法加载主数据")

        # Initialize normalization parameters for labels
        if self.y_mean is None or self.y_std is None:
            self.y_mean, self.y_std = get_norm_param(X=self.Y_norm, y_cols=self.y_var)

        if self.train:
            self.ws, self.stride = (ws, stride)
        else:
            self.ws, self.stride = (ws, 1)  # Stride for test is set to 1

        # Preprocessing data
        self.X = self.preprocess_data(
            self.X, norm_mode="z-score", ws=self.ws, stride=self.stride
        )
        self.Y = self.preprocess_labels(
            self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride
        )

        # Shuffling for training
        if self.train:
            np.random.seed(1234)
            if len(self.X) > 0:
                idx = np.random.permutation(len(self.X))
                self.X = self.X[idx]
                self.Y = self.Y[idx]

    def load_data(self, fermentation_number=None):
        """加载发酵数据"""
        # Load data for train/test fermentations
        X = []
        Y = []
        all_x_cols = set()  # 收集所有列名

        # Loading single fermentation data
        if fermentation_number is not None:
            data = load_data(
                work_dir=self.work_dir,
                fermentation_number=fermentation_number,
                y_cols=self.y_var
            )
            if data[0] is not None and data[1] is not None:
                X.append(data[0])
                Y.append(data[1])
                
                # 记录列名
                if data[0].shape[1] > 0:
                    all_x_cols.update([f"col_{i}" for i in range(data[0].shape[1])])

            return np.array(X), np.array(Y)

        # Loading train/test fermentations data
        fermentations = self.train_fermentations if self.train else self.test_fermentations
        
        for fn in fermentations:
            data = load_data(
                work_dir=self.work_dir,
                fermentation_number=fn,
                y_cols=self.y_var
            )
            if data[0] is not None and data[1] is not None:
                X.append(data[0])
                Y.append(data[1])
                
                # 记录列名
                if data[0].shape[1] > 0:
                    all_x_cols.update([f"col_{i}" for i in range(data[0].shape[1])])

        # 设置x_var为数字索引
        if not all_x_cols:
            print("警告: 没有找到特征列，使用默认列名")
            self.x_var = ["feature_" + str(i) for i in range(10)]  # 默认列名
        else:
            self.x_var = sorted(all_x_cols)
        
        # 如果指定了时间戳，筛选对应数据
        if self.timestamp is not None:
            X_filtered = []
            Y_filtered = []
            for x, y in zip(X, Y):
                if x is not None and y is not None and len(x) > 0:
                    # 由于我们不知道时间戳列的位置，假设第一列是时间戳
                    timestamps = x[:, 0]  # 假设时间戳在第一列
                    mask = timestamps == self.timestamp
                    if np.any(mask):
                        X_filtered.append(x[mask])
                        Y_filtered.append(y[mask])
            X = X_filtered
            Y = Y_filtered

        return np.array(X), np.array(Y)

    def preprocess_data(self, X, norm_mode="z-score", ws=20, stride=10):
        """预处理特征数据"""
        # Preprocess data
        if self.X_norm is None or len(self.X_norm) == 0:
            print("警告: 归一化数据为空，跳过归一化")
            mean = 0
            std = 1
        else:
            mean, std = get_norm_param(X=self.X_norm, x_cols=self.x_var)

        if self.saved_windows is None:
            processed_X = []
            for i, x in enumerate(X):
                if x is None or len(x) == 0:
                    continue
                    
                x = self.cumulative2snapshot(x)
                x = self.normalise(x, mean, std, norm_mode)
                sequences = self.data2sequences(x, ws, stride)
                if len(sequences) > 0:
                    processed_X.append(sequences)
            
            if processed_X:
                processed_X = np.concatenate(processed_X, axis=0)
                self.saved_windows = processed_X
                self.last_row_index = len(processed_X) - 1
            else:
                self.saved_windows = np.array([])
        return self.saved_windows

    def preprocess_labels(self, Y, norm_mode="z-score", ws=20, stride=10):
        """预处理标签数据"""
        # Preprocess labels
        if self.Y_norm is None or len(self.Y_norm) == 0:
            print("警告: 标签归一化数据为空，跳过归一化")
            mean = 0
            std = 1
        else:
            mean, std = get_norm_param(X=self.Y_norm, y_cols=self.y_var)

        processed_Y = []
        for y in Y:
            if y is None or len(y) == 0:
                continue
                
            # 确保是二维数组
            if y.ndim == 1:
                y = y.reshape(-1, 1)
                
            # 关键修复：跳过Y的累积处理
            if not self.skip_y_cumulative:
                y = self.cumulative2snapshot(y)
            
            y = self.normalise(y, mean, std, norm_mode)  # 归一化
            sequences = self.data2sequences(y, ws, stride)  # 转换为序列
            if len(sequences) > 0:
                processed_Y.append(sequences)

        if processed_Y:
            return np.concatenate(processed_Y, axis=0)
        return np.array([])

    def cumulative2snapshot(self, X):
        """将累积数据转换为快照数据"""
        if X is None or len(X) == 0:
            return X
            
        # 确保是二维数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        X = np.copy(X)

        # 如果没有设置x_var，跳过处理
        if self.x_var is None:
            return X

        for cv in self.cumulative_var:
            # 只处理X变量，跳过Y变量
            if cv in self.x_var:
                try:
                    idx = self.x_var.index(cv)
                    if idx < X.shape[1]:  # 防止索引越界
                        X[:, idx] = cumulative2snapshot(X[:, idx])
                except ValueError:
                    continue  # 如果变量不存在，跳过

        return X

    def normalise(self, X, mean, std, mode="z-score"):
        """归一化数据"""
        if X is None or len(X) == 0:
            return X
            
        # 确保是二维数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        binary_var_idx = []
        if self.x_var is not None:
            for bv in self.binary_var:
                if bv in self.x_var:
                    try:
                        idx = self.x_var.index(bv)
                        if idx < X.shape[1]:
                            binary_var_idx.append(idx)
                    except ValueError:
                        continue

        return normalise(
            X, mean=mean, std=std, mode=mode, binary_var=binary_var_idx
        )

    def data2sequences(self, X, ws=20, stride=10):
        """将数据转换为序列"""
        if X is None or len(X) < ws:
            return np.array([])
            
        # 确保是二维数组
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        sequences = []
        for i in range(0, len(X) - ws + 1, stride):
            sequences.append(X[i:i + ws])
        return np.array(sequences)

    def __getitem__(self, index):
        """获取单个样本"""
        if len(self.X) == 0 or index >= len(self.X):
            return [torch.zeros(self.ws, len(self.x_var)), torch.zeros(1)] if self.x_var else [torch.zeros(self.ws, 1), torch.zeros(1)]
        
        x = self.X[index]
        y = self.Y[index] if index < len(self.Y) else [0]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return [x, y]

    def __len__(self):
        """数据集大小"""
        return len(self.X)

    def get_num_features(self):
        """特征数量"""
        return self.X.shape[-1] if len(self.X) > 0 else 0

