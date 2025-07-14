import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import argparse
import numpy as np
from datasetC import *
import pdb
import warnings
from model import *
import random
import utils
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
from Ga import ParameterOptimizer
from filelock import Timeout, FileLock
from collections import deque
import random
import shutil  # 新增：用于文件复制操作

# 注意力增强的LSTM校正模块
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
    
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

# 强化学习智能体
class RLAgent:
    def __init__(self, state_dim):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=2000)
        
        self.policy_net = AttentionLSTM(state_dim, 32).cuda()
        self.target_net = AttentionLSTM(state_dim, 32).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(-0.1, 0.1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
                return self.policy_net(state_tensor).cpu().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).unsqueeze(1).cuda()
        actions = torch.FloatTensor(actions).cuda()
        rewards = torch.FloatTensor(rewards).cuda()
        if next_states[0] is not None:
            next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1).cuda()
        else:
            next_states = None
        dones = torch.FloatTensor(dones).cuda()
        
        current_q = self.policy_net(states).squeeze()
        
        # 处理终止状态
        with torch.no_grad():
            if next_states is None:
                next_q = torch.zeros_like(current_q)
            else:
                next_q = self.target_net(next_states).squeeze()
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= 0.995
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 校正系统
class ModelCorrector:
    """模型校正系统"""
    def __init__(self, base_model, state_dim, y_mean, y_std):
        self.base_model = base_model
        self.agent = RLAgent(state_dim=3*11 + 1)  # 最后3步特征+预测值
        self.y_mean = y_mean
        self.y_std = y_std
    
    def prepare_state(self, features, prediction):
        """构建强化学习状态向量"""
        recent_features = features[-3:].flatten()  # 仅使用最后3个时间步
        return np.concatenate([recent_features, [prediction]])
    
    def correct_prediction(self, features):
        """使用强化学习校正预测"""
        with torch.no_grad():
            h = self.base_model.init_hidden(1)
            input_tensor = torch.FloatTensor(features).unsqueeze(0).cuda()
            base_pred, _ = self.base_model(input_tensor, h)
            base_pred = base_pred[-1][-1].item()
        
        state = self.prepare_state(features, base_pred)
        correction = self.agent.select_action(state)
        return base_pred + correction

    def train_corrector(self, dataset, epochs=10, batch_size=32):
        """使用历史数据训练校正器"""
        # 预计算基础预测
        self.base_model.eval()
        base_preds = []
        for features, true_val, _ in dataset:
            with torch.no_grad():
                h = self.base_model.init_hidden(1)
                input_tensor = torch.FloatTensor(features).unsqueeze(0).cuda()
                pred, _ = self.base_model(input_tensor, h)
                #print(f"predpred: {pred}")
                # 加载归一化参数
                #y_mean = 26.282285714285713
                #y_std = 21.983558494047873
                preds_denorm1 = pred * self.y_std + self.y_mean
                initial_od6001 = preds_denorm1[-1][-1]
                #print(f"predpred2: {initial_od6001}")
                pred_value = initial_od6001.item()

                # 检查预测值是否有效
                if np.isnan(pred_value) or np.isinf(pred_value):
                    print(f"警告: 无效的基础预测值: {pred_value}")
                    pred_value = 0.0  # 设为默认值
                
                base_preds.append(pred_value)
        
        # 强化学习训练
        self.agent.policy_net.train()
        for epoch in range(epochs):
            states, actions, rewards = [], [], []
            valid_samples = 0
            
            for idx, (features, true_val, _) in enumerate(dataset):
                # 跳过无效数据点
                if idx >= len(base_preds):
                    continue
                    
                if np.isnan(true_val) or np.isinf(true_val):
                    print(f"跳过无效真实值: {true_val}")
                    continue
                    
                state = self.prepare_state(features, base_preds[idx])
                state = state.astype(np.float32)  # 确保数据类型
                
                # 检查状态是否有效
                if np.isnan(state).any() or np.isinf(state).any():
                    print(f"跳过无效状态: {state}")
                    continue
                    
                action = self.agent.select_action(state)
                corrected_pred = base_preds[idx] + action
                
                # 计算奖励：负绝对误差，添加小值防止NaN
                error = abs(corrected_pred - true_val)
                if np.isnan(error) or error == 0:
                    error = 1e-6  # 防止除以零或NaN
                    
                reward = -error
                rewards.append(reward)
                
                # 存储经验（单步终止）
                self.agent.remember(state, action, reward, None, True)
                
                # 记录批次数据
                states.append(state)
                actions.append(action)
                valid_samples += 1
            
            if valid_samples == 0:
                print(f"Epoch {epoch+1} | 无有效样本")
                continue
                
            # 批量经验回放
            self.agent.replay(batch_size)
            
            # 计算平均奖励
            avg_reward = np.mean(rewards)
            print(f"Epoch {epoch+1} | Avg Reward: {avg_reward:.4f} | 有效样本: {valid_samples}")
            
            # 更新目标网络
            if epoch % 5 == 0:
                self.agent.update_target()
        self.agent.policy_net.eval()


# 数据准备函数
def load_correction_data(data_path, pred_path, seq_length=20):
    """从Excel文件加载校正训练数据"""
    # 读取原始数据
    data_df = pd.read_excel(data_path, dtype={'Timestamp': str})
    pred_df = pd.read_excel(pred_path, dtype={'Timestamp': str})
    
    # 确保没有前导/尾随空格
    data_df['Timestamp'] = data_df['Timestamp'].str.strip()
    pred_df['Timestamp'] = pred_df['Timestamp'].str.strip()
    #print(f"true_value: {data_df}")
    #print(f"true_value: {pred_df}")
    # 按时间戳合并
    merged_df = pd.merge(data_df, pred_df, on="Timestamp", suffixes=('_true', '_pred'))
    #print(f"true_value: {merged_df}")
    # 提取特征和标签 - 使用正确的后缀
    feature_columns = [
        'dm_air_true', 'm_ls_opt_do_true', 'm_ph_true', 'm_stirrer_true', 'm_temp_true',
        'dm_o2_true', 'dm_spump1_true', 'dm_spump2_true', 'dm_spump3_true', 'dm_spump4_true', 'induction_true'
    ]
    
    features = merged_df[feature_columns].values
    true_od = merged_df['od_600_true'].values
    pred_od = merged_df['od_600_pred'].values
    #print(f"true_value: {features}")
    # 构建数据集
    dataset = []
    valid_count = 0
    for i in range(seq_length, len(features)):
        # 当前序列
        seq_features = features[i-seq_length:i]
        true_value = true_od[i]
        #print(f"true_value: {true_value}")
        #print(f"seq_features: {seq_features}")
        # 跳过无效数据点
        if np.isnan(seq_features).any() or np.isnan(true_value):
            print(f"跳过无效数据点 (索引 {i})")
            continue
            
        dataset.append((seq_features, true_value, None))
        valid_count += 1
    
    print(f"加载校正数据: 共 {len(features)} 个点, 有效 {valid_count} 个")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--batch_size", default=256, type=int, help="test batchsize")
    parser.add_argument("--hidden_dim", default=16, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--seed", default=123)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--model", default="lstm", type=str)
    parser.add_argument("--pred_steps", default=1, type=int, help="Number of prediction steps")
    parser.add_argument("--train_corrector", action="store_true", help="是否训练强化学习校正器")
    parser.add_argument("--corrector_epochs", type=int, default=50, help="校正器训练轮数")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 创建结果文件
    result_file = "predictseek.xlsx"
    result_columns = [
    "Timestamp", "dm_air", "m_ls_opt_do", "m_ph", "m_stirrer", "m_temp",
    "dm_o2", "dm_spump1", "dm_spump2", "dm_spump3", "dm_spump4", "induction",
    "od_600", "RMSE", "REFY"
    ]
    
    # 初始化结果文件
    if not os.path.exists(result_file):
        pd.DataFrame(columns=result_columns).to_excel(result_file, index=False)
    
    # 获取数据文件路径
    data_file_path = os.path.join(args.dataset, "28", "data.xlsx")
    #############################################################
    # 优化：确保文件操作后及时释放权限
    #############################################################
    print("等待数据文件满足条件（Timestamp列最后一个值 >= 100）...")
    while True:
        try:
            # 临时变量存储数据，确保快速释放文件
            df = None
            lock_path = data_file_path + ".lock"
            
            # 使用文件锁确保安全读取
            with FileLock(lock_path, timeout=10):
                # 读取数据文件
                df = pd.read_excel(data_file_path)
            
            # 在锁外处理数据，减少锁持有时间
            if df is not None and not df.empty:
                # 获取Timestamp列最后一个值
                last_timestamp = df['Timestamp'].iloc[-1]
                
                # 处理可能的字符串类型时间戳
                if isinstance(last_timestamp, str):
                    try:
                        last_timestamp = float(last_timestamp)
                    except ValueError:
                        last_timestamp = 0
                
                print(f"当前Timestamp最后值: {last_timestamp}")
                
                # 检查是否满足条件
                if last_timestamp >= 100:
                    print("数据文件满足条件，继续执行程序...")
                    break
            
            # 复制文件到根目录（在锁外操作）
            shutil.copy(data_file_path, result_file)
            print(f"已将数据文件复制到: {result_file}")
            
            # 立即释放数据框内存
            del df
            df = None
            
        except (Timeout, BlockingIOError):
            print("文件被占用，等待10秒后重试...")
        except Exception as e:
            print(f"处理数据文件时出错: {str(e)}")
        
        # 等待10秒
        time.sleep(10)
    # 记录最后修改时间
    last_mod_time = os.path.getmtime(data_file_path)
    
    # 加载归一化参数
    norm_file_path = os.path.join(args.dataset, "norm_file.json")
    with open(norm_file_path, 'r') as f:
        norm_data = json.load(f)
    y_mean = norm_data['y_mean']
    y_std = norm_data['y_std']
    #print(f"true_valuey_std: {y_mean, y_std}")
    # 设置模型
    if args.model == "lstm":
        model = LSTMPredictor(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        ).cuda()
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=11,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        ).cuda()
    # 加载模型权重
    weights = (
        os.path.join("model_weights", "od600_prediction", args.model, "weights_best.tar")
        if args.weights == ""
        else args.weights
    )
    model = utils.load_weights(model, weights)
    model.eval()  # ⭐ 新增：设置为评估模式
    mse = nn.MSELoss()
    
    # 初始化校正器
    corrector = ModelCorrector(base_model=model, state_dim=34, y_mean=y_mean, y_std=y_std)
    
    # 训练校正器（如果指定）
    if args.train_corrector:
        print("\n===== 训练强化学习校正器 =====")
        correction_data = load_correction_data(
            data_path=os.path.join(args.dataset, "28", "data.xlsx"),
            pred_path="predictseek.xlsx",
            seq_length=20
        )
        corrector.train_corrector(correction_data, epochs=args.corrector_epochs)
        torch.save(corrector.agent.policy_net.state_dict(), "rl_corrector.pth")
        print("校正器模型已保存至 rl_corrector.pth")
    
    # 加载训练好的校正器
    if os.path.exists("rl_corrector.pth"):
        corrector.agent.policy_net.load_state_dict(torch.load("rl_corrector.pth"))
        corrector.agent.policy_net.eval()
        print("已加载预训练校正器")

    # Predict with overlapped sequences
    def test(epoch, model, testloader, corrector=None):
        model.eval()
        loss = 0
        err = 0
        iter = 0
        h = model.init_hidden(args.batch_size)
        preds = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        labels = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        n_overlap = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        N = 10
        
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(testloader):
                iter += 1
                batch_size = input.size(0)
                h = model.init_hidden(batch_size)
                h = tuple([e.data.cuda() for e in h])
                input, label = input.cuda(), label.cuda()
                #print(f"seq_features: {corrector}")
                # 使用校正预测器 - 修复此处
                if corrector:
                    # 确保输入形状正确 [batch_size, seq_len, features]
                    if input.dim() == 2:
                        # 处理单样本情况
                        input = input.unsqueeze(0)
                    
                    y = []
                    #print(f"seq_features: {y}")
                    for i in range(input.size(0)):  # 遍历batch中的每个样本
                        # 获取单个序列 (seq_len, features)
                        seq = input[i]
                        # 将序列转换为numpy数组
                        seq_np = seq.cpu().numpy()
                        #print(f"seq_np: {seq_np[-1]}")
                        # 使用ModelCorrector进行校正
                        corrected_pred = corrector.correct_prediction(seq_np)
                        #print(f"seq_features: {corrected_pred}")
                        y.append(corrected_pred)
                    y = np.array(y)
                else:
                    output, h = model(input.float().cuda(), h)
                    print(f"seq_features111111: {output}")
                    y = output.view(-1).cpu().numpy()
                #print(f"seq_features1111011: {y}")    
                #output, h = model(input.float().cuda(), h)
                #print(f"seq_features111111: {output}")
                
                
                #y = output.view(-1).cpu().numpy()
                #print(f"seq_features111111: {y}")
                # 确保y是标量数组

                
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
                
                preds[batch_idx: (batch_idx + test_dataset.ws)] += y_smooth
                labels[batch_idx: (batch_idx + test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx: (batch_idx + test_dataset.ws)] += 1.0
                

                #print(f"1111: {preds}")
                # 计算损失
                y_tensor = torch.tensor([y]).float().cuda()  # 创建包含单个值的张量
                loss += nn.MSELoss()(y_tensor, label.float().cuda())
                err += torch.sqrt(nn.MSELoss()(y_tensor, label.float().cuda())).item()
        loss = loss / len(test_dataset)
        err = err / len(test_dataset)
        preds /= n_overlap
        #print(f"2222: {preds}")
        labels /= n_overlap
        return err, preds, labels

    def _normalize_individual(optimized_params):
        """
        归一化 individual 数据
        """
        return (optimized_params - x_mean) / x_std
    

    # 加载归一化参数
    norm_file_path = os.path.join(args.dataset, "norm_file.json")
    with open(norm_file_path, 'r') as f:
        norm_data = json.load(f)
    y_mean = norm_data['y_mean']
    y_std = norm_data['y_std']
    

    # 设置模型
    if args.model == "lstm":
        model = LSTMPredictor(
            input_dim=11,  # 直接指定输入维度（与数据集无关）
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        ).cuda()
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=11,  # 直接指定输入维度（与数据集无关）
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        ).cuda()
    # 加载模型权重
    weights = (
        os.path.join("model_weights", "od600_prediction", args.model, "weights_best.tar")
        if args.weights == ""
        else args.weights
    )
    model = utils.load_weights(model, weights)
    mse = nn.MSELoss()
    
    # 初始化行数计数器
    result_file = "predictseek.xlsx"
    INITIAL_ROW_COUNT = 0
    if os.path.exists(result_file):
        df = pd.read_excel(result_file)
        INITIAL_ROW_COUNT = len(df)
    
    # 训练阈值设置
    TRAIN_THRESHOLD = 2  # 每增加60行数据训练一次
    # 主监控循环
    while True:
        try:
            # 检查文件是否更新
            current_mod_time = os.path.getmtime(data_file_path)
            if current_mod_time > last_mod_time:
                print(f"\n检测到数据更新: {time.ctime(current_mod_time)}")
                last_mod_time = current_mod_time
                
                # 尝试读取数据文件
                data_loaded = False
                while not data_loaded:
                    try:
                        # 使用文件锁确保安全读取
                        with FileLock(data_file_path + ".lock"):
                            test_dataset = FermentationData(
                                work_dir=args.dataset, 
                                train_mode=False, 
                                y_var=["od_600"],
                                y_mean=y_mean,
                                y_std=y_std
                            )
                        data_loaded = True
                    except (Timeout, BlockingIOError):
                        print("数据文件被占用，1秒后重试...")
                        time.sleep(1)
                
                # 获取 X 的归一化参数
                x_mean, x_std = utils.get_norm_param(X=test_dataset.X_norm, x_cols=test_dataset.x_var)
                
                # 设置DataLoader
                test_loader = torch.utils.data.DataLoader(
                    dataset=test_dataset, batch_size=1, num_workers=0, shuffle=False
                )
                
                # 参数优化相关初始化
                param_names = test_dataset.x_var
                initial_state = test_dataset.X[-1][-1].copy() * x_std + x_mean
                initial_params = initial_state[:11].copy()
                print(f"Initial params: {initial_params}")
                
                epsilon = 1e-6
                
                # 调整参数优化范围
                param_bounds = [
                    (max(epsilon, initial_params[0] - 0), initial_params[0] + initial_params[0]*0.1),
                    (max(epsilon, initial_params[1] - initial_params[1]*0.1), initial_params[1] + initial_params[1]*0.1),
                    (max(3.0, initial_params[2] - initial_params[2]*0.1), min(8.0, initial_params[2] + initial_params[2]*0.1)),
                    (max(epsilon, initial_params[3] - initial_params[3]*0.1), initial_params[3] + initial_params[3]*0.1),
                    (initial_params[4] - initial_params[4]*0.1, initial_params[4] + initial_params[4]*0.1),
                    (max(epsilon, initial_params[5] - 0), initial_params[5] + initial_params[5]*0.1),
                    (max(epsilon, initial_params[6] - 0), initial_params[6] + initial_params[6]*0.1),
                    (max(epsilon, initial_params[7] - 0), initial_params[7] + initial_params[7]*0.1),
                    (max(epsilon, initial_params[8] - 0), initial_params[8] + initial_params[8]*0.1),
                    (max(epsilon, initial_params[9] - 0), initial_params[9] + initial_params[9]*0.1),
                    (max(0.1, initial_params[10] - initial_params[10]*0.1), min(0.9, initial_params[10] + initial_params[5]*0.1))
                ]
                
                # 第一轮测试
                print("\nInitial Testing")
                err, preds, labels = test(0, model, test_loader, corrector)
                preds_denorm = preds * y_std + y_mean
                initial_od600 = preds_denorm[-1]
                print(f"Initial predict OD600: {initial_od600}")
                
                # 参数优化
                optimizer = ParameterOptimizer(param_names, param_bounds, model, 
                                              test_dataset, x_mean, x_std, y_mean, y_std)
                optimized_params = test_dataset.X[-1][-1] * x_std + x_mean  # 逆归一化得到初始参数值
                best_od600 = initial_od600
                best_params = optimized_params.copy()
                
                # 执行优化
                optimized_params = optimizer.optimize(initial_params, best_od600)
                
                # 更新数据集参数
                optimized_params_norm = _normalize_individual(optimized_params)
                test_dataset.X[-1][-1] = optimized_params_norm
                
                # 重新测试
                err, preds, labels = test(0, model, test_loader, corrector)
                preds_denorm = preds * y_std + y_mean
                current_od600 = preds_denorm[-1]
                
                # 更新最佳结果
                if current_od600 > best_od600:
                    best_od600 = current_od600
                    best_params = optimized_params
                
                # 获取 data.xlsx 中的 Timestamp 值
                try:
                    df_data = pd.read_excel(data_file_path)
                    timestamp = df_data.iloc[-1, 0]  # 读取最后一行的 A 列时间戳
                except Exception as e:
                    print(f"读取 Timestamp 时发生错误: {str(e)}")
                    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 逆归一化预测结果和标签
                preds_denorm = preds * y_std + y_mean
                labels_denorm = labels * y_std + y_mean
                preds_denorm = preds_denorm.reshape(-1, 1)
                labels_denorm = labels_denorm.reshape(-1, 1)
                preds_denorm = preds_denorm[50:]
                labels_denorm = labels_denorm[50:]
                mse = np.square(np.subtract(preds_denorm, labels_denorm)).mean()
                rmse = math.sqrt(mse)
                
                # Relative Error on Final Yield
                refy = abs(preds_denorm[-1] - labels_denorm[-1]) / labels_denorm[-1] * 100
                
                # 保存结果
                result_saved = False
                retry_count = 0
                max_retries = 20  # 最大重试次数
                while not result_saved and retry_count < max_retries:
                    try:
                        # 读取现有数据
                        if os.path.exists(result_file):
                            df = pd.read_excel(result_file)
                        else:
                            df = pd.DataFrame(columns=result_columns)
                        
                        # 添加新结果
                        new_row = {
                            "Timestamp": timestamp,
                            "dm_air": best_params[0],
                            "m_ls_opt_do": best_params[1],
                            "m_ph": best_params[2],
                            "m_stirrer": best_params[3],
                            "m_temp": best_params[4],
                            "dm_o2": best_params[5],
                            "dm_spump1": best_params[6],
                            "dm_spump2": best_params[7],
                            "dm_spump3": best_params[8],
                            "dm_spump4": best_params[9],
                            "induction": best_params[10],
                            "od_600": current_od600,
                            "RMSE": rmse,
                            "REFY": refy
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                        
                        # 保存更新
                        df.to_excel(result_file, index=False)
                        print(f"结果已保存至 {result_file}")
                        result_saved = True
                    except PermissionError:
                        print("结果文件被占用，1秒后重试...")
                        time.sleep(1)
                        retry_count += 1
                
                if not result_saved:
                    print(f"警告: 已达到最大重试次数({max_retries})，未能保存结果")

                print("\nOptimization Results:")
                print(f"Initial predict OD600: {initial_od600}, Initial parameters: {initial_params}")
                print(f"Best OD600: {best_od600}, Best parameters: {best_params}")
                data_path = os.path.join(args.dataset, "28", "data.xlsx")
                df = pd.read_excel(data_path)
                first_column = df.iloc[:, 0].values

                # 计算最小公共长度
                min_length = min(
                    len(first_column[50:]), 
                    len(preds_denorm),
                    len(labels_denorm)
                )

                # 统一截取长度
                first_column = first_column[50:][:min_length]
                preds_denorm = preds_denorm[:min_length]
                labels_denorm = labels_denorm[:min_length]
                rmse_array = np.full(min_length, rmse)
                refy_array = np.full(min_length, refy)
                
                # 将第一列数据添加到 results.npz 的最左侧
                timestamp_data = np.column_stack((first_column, preds_denorm, labels_denorm, rmse_array, refy_array))
                
                # 保存到 results.npz
                np.savez(
                    weights.split("/weights")[0] + "/results.npz",
                    Timestamp=timestamp_data,
                    preds=preds_denorm,
                    labels=labels_denorm,
                    rmse=rmse,
                    refy=refy,
                )
                
                print("Saved: ", weights.split("/weights")[0] + "/results.npz")
                print("\nRMSE Error OD600: ", rmse)
                print("\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1])))
                
                current_row_count = len(pd.read_excel(result_file))
                new_rows_added = current_row_count - INITIAL_ROW_COUNT
                # 检查是否达到训练阈值
                if new_rows_added >= TRAIN_THRESHOLD:
                    print(f"\n===== 新增{new_rows_added}行数据，开始训练校正器 =====")
                    
                    # 加载校正数据（使用最新的predictseek.xlsx）
                    correction_data = load_correction_data(
                        data_path=os.path.join(args.dataset, "28", "data.xlsx"),
                        pred_path=result_file,  # 使用当前结果文件
                        seq_length=20
                    )
                    
                    # 训练校正器
                    corrector.train_corrector(correction_data, epochs=args.corrector_epochs)
                    
                    # 保存并重新加载校正器模型
                    torch.save(corrector.agent.policy_net.state_dict(), "rl_corrector.pth")
                    corrector.agent.policy_net.load_state_dict(torch.load("rl_corrector.pth"))
                    corrector.agent.policy_net.eval()
                    print(f"校正器模型已更新（基于{current_row_count}行数据）")
                    
                    # 更新初始行数基准
                    INITIAL_ROW_COUNT = current_row_count                
                # 读取 data.xlsx 文件的第一列数据

            
            # 等待30秒再次检查
            time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            time.sleep(3)  # 出错后等待30秒重试
    

    # 读取 data.xlsx 文件的第一列数据
    data_path = os.path.join(args.dataset, "28", "data.xlsx")
    df = pd.read_excel(data_path)
    first_column = df.iloc[:, 0].values
    first_column = first_column[50:]  # 与 preds 和 labels 对齐
    
    # 将 rmse 和 refy 扩展为与 first_column 长度相同的数组
    rmse_array = np.full_like(first_column, rmse)
    refy_array = np.full_like(first_column, refy)
    
    # 将第一列数据添加到 results.npz 的最左侧
    timestamp_data = np.column_stack((first_column, preds_denorm, labels_denorm, rmse_array, refy_array))
    
    # 保存到 results.npz
    np.savez(
        weights.split("/weights")[0] + "/results.npz",
        Timestamp=timestamp_data,
        preds=preds_denorm,
        labels=labels_denorm,
        rmse=rmse,
        refy=refy,
    )
    print("Saved: ", weights.split("/weights")[0] + "/results.npz")
    
    # 绘制曲线
    utils.plot_od600_curve(
        preds_denorm, labels_denorm, weights[:-17], rmse, refy
    )
    print("\nRMSE Error OD600: ", rmse)
    print("\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1])))
    
    # 绘制多步预测曲线
    plt.figure(figsize=(10, 6))
    plt.title(f"Multi-step Prediction (Prediction Steps: {args.pred_steps})")
    plt.plot(labels_denorm, label="True Values", color="black", linewidth=2)
    for step in range(args.pred_steps):
        plt.plot(preds_denorm[:, step], label=f"Predicted Step {step + 1}")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("OD600")
    plt.savefig(weights.split("/weights")[0] + "/multi_step_prediction.png")
    plt.show()


if __name__ == '__main__':
    main()