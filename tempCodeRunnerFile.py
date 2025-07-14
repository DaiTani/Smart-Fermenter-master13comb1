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
import pymysql
from sqlalchemy import create_engine
import pymysql.cursors
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

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

# 数据库操作类
class MySQLDatabase:
    def __init__(self, host, user, password, database, port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
        
    def query(self, sql):
        """执行SQL查询并返回结果"""
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
            return result
        finally:
            connection.close()
    
    def insert(self, table, data):
        """插入数据到表中"""
        connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port
        )
        try:
            with connection.cursor() as cursor:
                # 构建插入语句
                columns = ', '.join(data.keys())
                placeholders = ', '.join(['%s'] * len(data))
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, list(data.values()))
            connection.commit()
        finally:
            connection.close()
    
    def get_last_timestamp(self, table):
        """获取表中最后一条记录的时间戳"""
        result = self.query(f"SELECT Timestamp FROM {table} ORDER BY Timestamp DESC LIMIT 1")
        if result:
            return result[0]['Timestamp']
        return None
    
    def get_row_count(self, table):
        """获取表中的行数"""
        result = self.query(f"SELECT COUNT(*) AS count FROM {table}")
        if result:
            return result[0]['count']
        return 0
    
    def get_table_data(self, table):
        """获取整个表的数据"""
        return self.query(f"SELECT * FROM {table}")
    
    def save_dataframe(self, df, table, if_exists='append'):
        """将DataFrame保存到数据库表"""
        df.to_sql(table, self.engine, if_exists=if_exists, index=False)

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
                preds_denorm1 = pred * self.y_std + self.y_mean
                initial_od6001 = preds_denorm1[-1][-1]
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
def load_correction_data(db, data_table, pred_table, seq_length=20):
    """从MySQL数据库加载校正训练数据"""
    # 读取原始数据
    data_query = f"SELECT * FROM {data_table}"
    pred_query = f"SELECT * FROM {pred_table}"
    
    data_df = pd.DataFrame(db.query(data_query))
    pred_df = pd.DataFrame(db.query(pred_query))
    
    # 动态获取标签
    all_columns = data_df.columns.tolist()
    x_columns = [col for col in all_columns if col not in ["Timestamp", "od_600"]]
    
    # 按时间戳合并
    merged_df = pd.merge(data_df, pred_df, on="Timestamp", suffixes=('_true', '_pred'))
    
    feature_columns = [f"{col}_true" for col in x_columns]
    features = merged_df[feature_columns].values 
    true_od = merged_df['od_600_true'].values
    pred_od = merged_df['od_600_pred'].values
    
    # 构建数据集
    dataset = []
    valid_count = 0
    for i in range(seq_length, len(features)):
        # 当前序列
        seq_features = features[i-seq_length:i]
        true_value = true_od[i]
        
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
    parser.add_argument("--corrector_epochs", type=int, default=10, help="校正器训练轮数")
    
    # 数据库配置参数
    parser.add_argument("--db_host", default="localhost", type=str, help="MySQL host")
    parser.add_argument("--db_user", default="root", type=str, help="MySQL user")
    parser.add_argument("--db_password", default="password", type=str, help="MySQL password")
    parser.add_argument("--db_name", default="fermentation", type=str, help="MySQL database name")
    parser.add_argument("--db_port", default=3306, type=int, help="MySQL port")
    parser.add_argument("--data_table", default="data_table", type=str, help="原始数据表名")
    parser.add_argument("--result_table", default="result_table", type=str, help="结果表名")
    
    args = parser.parse_args()
    
    # 初始化数据库连接
    db = MySQLDatabase(
        host=args.db_host,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        port=args.db_port
    )
    
    # 创建结果表（如果不存在）
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {args.result_table} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Timestamp FLOAT,
        dm_air FLOAT,
        m_ls_opt_do FLOAT,
        m_ph FLOAT,
        m_stirrer FLOAT,
        m_temp FLOAT,
        dm_o2 FLOAT,
        dm_spump1 FLOAT,
        dm_spump2 FLOAT,
        dm_spump3 FLOAT,
        dm_spump4 FLOAT,
        induction FLOAT,
        od_600 FLOAT,
        RMSE FLOAT,
        REFY FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    db.query(create_table_sql)
    
    torch.cuda.set_device(args.gpuid)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 获取数据列名
    data_df = pd.DataFrame(db.query(f"SELECT * FROM {args.data_table} LIMIT 1"))
    all_columns = data_df.columns.tolist()
    x_columns = [col for col in all_columns if col not in ["Timestamp", "od_600"]]
    n_features = len(x_columns)
    
    # 加载归一化参数
    norm_file_path = os.path.join(args.dataset, "norm_file.json")
    with open(norm_file_path, 'r') as f:
        norm_data = json.load(f)
    y_mean = norm_data['y_mean']
    y_std = norm_data['y_std']
    
    # 设置模型
    if args.model == "lstm":
        model = LSTMPredictor(
            input_dim=n_features,
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        ).cuda()
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=n_features,
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
    model.eval()
    mse = nn.MSELoss()
    
    # 初始化校正器
    corrector = ModelCorrector(
        base_model=model, 
        state_dim=3 * n_features + 1,
        y_mean=y_mean, 
        y_std=y_std
    )
    
    # 训练校正器（如果指定）
    if args.train_corrector:
        print("\n===== 训练强化学习校正器 =====")
        correction_data = load_correction_data(
            db=db,
            data_table=args.data_table,
            pred_table=args.result_table,
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
                
                # 使用校正预测器
                if corrector:
                    if input.dim() == 2:
                        input = input.unsqueeze(0)
                    
                    y = []
                    for i in range(input.size(0)):
                        seq = input[i]
                        seq_np = seq.cpu().numpy()
                        corrected_pred = corrector.correct_prediction(seq_np)
                        y.append(corrected_pred)
                    y = np.array(y)
                else:
                    output, h = model(input.float().cuda(), h)
                    y = output.view(-1).cpu().numpy()
                
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
                
                preds[batch_idx: (batch_idx + test_dataset.ws)] += y_smooth
                labels[batch_idx: (batch_idx + test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx: (batch_idx + test_dataset.ws)] += 1.0
                
                # 计算损失
                y_tensor = torch.tensor([y]).float().cuda()
                loss += nn.MSELoss()(y_tensor, label.float().cuda())
                err += torch.sqrt(nn.MSELoss()(y_tensor, label.float().cuda())).item()
        loss = loss / len(test_dataset)
        err = err / len(test_dataset)
        preds /= n_overlap
        labels /= n_overlap
        return err, preds, labels

    def _normalize_individual(optimized_params):
        """归一化 individual 数据"""
        return (optimized_params - x_mean) / x_std
    
    # 初始化行数计数器
    INITIAL_ROW_COUNT = db.get_row_count(args.result_table)
    
    # 训练阈值设置
    TRAIN_THRESHOLD = 2
    
    # 主监控循环
    last_timestamp = db.get_last_timestamp(args.data_table)
    
    while True:
        try:
            # 检查是否有新数据
            current_timestamp = db.get_last_timestamp(args.data_table)
            if current_timestamp != last_timestamp:
                print(f"\n检测到新数据: {current_timestamp}")
                last_timestamp = current_timestamp
                
                # 加载数据
                test_dataset = FermentationData(
                    work_dir=args.dataset, 
                    train_mode=False, 
                    y_var=["od_600"],
                    y_mean=y_mean,
                    y_std=y_std
                )
                
                # 获取 X 的归一化参数
                x_mean, x_std = utils.get_norm_param(X=test_dataset.X_norm, x_cols=test_dataset.x_var)
                
                # 设置DataLoader
                test_loader = torch.utils.data.DataLoader(
                    dataset=test_dataset, batch_size=1, num_workers=0, shuffle=False
                )
                
                # 参数优化相关初始化
                param_names = test_dataset.x_var
                initial_state = test_dataset.X[-1][-1].copy() * x_std + x_mean
                initial_params = initial_state[:n_features].copy()
                print(f"Initial params: {initial_params}")
                
                epsilon = 1e-6
                
                # 设置参数边界
                param_bounds = []
                for i in range(n_features):
                    feature_name = x_columns[i]
                    if feature_name == "m_ph":
                        param_bounds.append((3.0, 8.0))
                    elif feature_name == "induction":
                        param_bounds.append((0.1, 0.9))
                    else:
                        low = max(epsilon, initial_params[i] - initial_params[i]*0.1)
                        high = initial_params[i] + initial_params[i]*0.1
                        param_bounds.append((low, high))
                
                # 第一轮测试
                print("\nInitial Testing")
                err, preds, labels = test(0, model, test_loader, corrector)
                preds_denorm = preds * y_std + y_mean
                initial_od600 = preds_denorm[-1]
                print(f"Initial predict OD600: {initial_od600}")
                
                # 参数优化
                optimizer = ParameterOptimizer(param_names, param_bounds, model, 
                                              test_dataset, x_mean, x_std, y_mean, y_std)
                optimized_params = test_dataset.X[-1][-1] * x_std + x_mean
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
                
                # 获取时间戳
                timestamp = db.get_last_timestamp(args.data_table)
                
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
                
                # 保存结果到数据库
                result_data = {
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
                
                db.insert(args.result_table, result_data)
                print(f"结果已保存至数据库表 {args.result_table}")
                
                print("\nOptimization Results:")
                print(f"Initial predict OD600: {initial_od600}, Initial parameters: {initial_params}")
                print(f"Best OD600: {best_od600}, Best parameters: {best_params}")
                
                # 保存到 results.npz
                np.savez(
                    weights.split("/weights")[0] + "/results.npz",
                    preds=preds_denorm,
                    labels=labels_denorm,
                    rmse=rmse,
                    refy=refy,
                )
                print("Saved: ", weights.split("/weights")[0] + "/results.npz")
                print("\nRMSE Error OD600: ", rmse)
                print("\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1])))
                
                # 绘制曲线
                utils.plot_od600_curve(
                    preds_denorm, labels_denorm, weights[:-17], rmse, refy
                )
                
                # 检查是否需要训练校正器
                current_row_count = db.get_row_count(args.result_table)
                new_rows_added = current_row_count - INITIAL_ROW_COUNT
                if new_rows_added >= TRAIN_THRESHOLD:
                    print(f"\n===== 新增{new_rows_added}行数据，开始训练校正器 =====")
                    
                    # 加载校正数据
                    correction_data = load_correction_data(
                        db=db,
                        data_table=args.data_table,
                        pred_table=args.result_table,
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
            
            # 等待30秒再次检查
            time.sleep(30)
        
        except KeyboardInterrupt:
            print("\n监控已停止")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            time.sleep(30)  # 出错后等待30秒重试

if __name__ == '__main__':
    main()
