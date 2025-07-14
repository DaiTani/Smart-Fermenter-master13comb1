import sys
import time  # 新增时间模块
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
from Ga import ParameterOptimizer  # 导入优化器类
from filelock import Timeout, FileLock  # 用于文件锁管理

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
    
    # 记录最后修改时间
    last_mod_time = os.path.getmtime(data_file_path)

    # Predict with overlapped sequences
    def test(epoch, model, testloader):
        model.eval()
        loss = 0
        err = 0
        iter = 0
        # Initialise model hidden state
        h = model.init_hidden(args.batch_size)
        # Initialise vectors to store predictions, labels
        preds = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        labels = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        n_overlap = np.zeros(len(test_dataset) + test_dataset.ws - 1)
        N = 10
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(testloader):
                iter += 1
                batch_size = input.size(0)
                h = model.init_hidden(batch_size)
                h = tuple([e.data for e in h])
                input, label = input.cuda(), label.cuda()
                output, h = model(input.float(), h)
                y = output.view(-1).cpu().numpy()
                y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
                y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
                preds[batch_idx: (batch_idx + test_dataset.ws)] += y_smooth
                labels[batch_idx: (batch_idx + test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx: (batch_idx + test_dataset.ws)] += 1.0
                loss += nn.MSELoss()(output, label.float())
                err += torch.sqrt(nn.MSELoss()(output, label.float())).item()
        loss = loss / len(test_dataset)
        err = err / len(test_dataset)
        # Compute the average dividing for the number of overlaps
        preds /= n_overlap
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
            #input_dim=len(Data.x_var),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        )
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=11,  # 直接指定输入维度（与数据集无关）
            #input_dim=len(Data.x_var),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        )
    model.cuda()

    # 加载模型权重
    weights = (
        os.path.join("model_weights", "od600_prediction", args.model, "weights_best.tar")
        if args.weights == ""
        else args.weights
    )
    model = utils.load_weights(model, weights)
    mse = nn.MSELoss()

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
                err, preds, labels = test(0, model, test_loader)
                preds_denorm = preds * y_std + y_mean
                initial_od600 = preds_denorm[-1]
                print(f"Initial OD600: {initial_od600}")

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
                err, preds, labels = test(0, model, test_loader)
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
                # 获取当前时间戳
                #timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                # 保存结果
                result_saved = False
                retry_count = 0
                max_retries = 10  # 最大重试次数
                
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
                print(f"Initial OD600: {initial_od600}, Initial parameters: {initial_params}")
                print(f"Best OD600: {best_od600}, Best parameters: {best_params}")        
                

                
                
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



                print("\nRMSE Error OD600: ", rmse)
                print(
                    "\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1]))
                )
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
    print(
        "\nREFY: %.2f%%" % (refy), "[absolute error: %.2f]" % (abs(preds_denorm[-1] - labels_denorm[-1]))
    )

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

