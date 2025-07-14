import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
import numpy as np
from datasetC import *
import warnings
from model import *
import random
import utils
import math
import pandas as pd
import matplotlib.pyplot as plt
import json
from Ga import ParameterOptimizer
import time
from filelock import Timeout, FileLock
from datetime import datetime

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

    # Predict with overlapped sequences
    def test(epoch, model, testloader):
        model.eval()
        loss = 0
        err = 0
        iter = 0

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

                preds[batch_idx : (batch_idx + test_dataset.ws)] += y_smooth
                labels[batch_idx : (batch_idx + test_dataset.ws)] += label.view(-1).cpu().numpy()
                n_overlap[batch_idx : (batch_idx + test_dataset.ws)] += 1.0

                loss += nn.MSELoss()(output, label.float())
                err += torch.sqrt(nn.MSELoss()(output, label.float())).item()

        preds /= n_overlap
        labels /= n_overlap
        return err, preds, labels

    def _normalize_individual(optimized_params):
        return (optimized_params - x_mean) / x_std

    # 加载归一化参数
    norm_file_path = os.path.join(args.dataset, "norm_file.json")
    with open(norm_file_path, 'r') as f:
        norm_data = json.load(f)
    y_mean = norm_data['y_mean']
    y_std = norm_data['y_std']

    # 初始化结果文件
    result_file = "predictseek.xlsx"
    result_columns = ["Timestamp", "Optimized_OD600", "Best_Parameters", "RMSE", "REFY"]
    if not os.path.exists(result_file):
        pd.DataFrame(columns=result_columns).to_excel(result_file, index=False)

    # 数据文件路径
    data_file_path = os.path.join(args.dataset, "28", "data.xlsx")
    
    # 设置模型
    if args.model == "lstm":
        model = LSTMPredictor(
            input_dim=len(test_dataset.x_var),
            hidden_dim=args.hidden_dim,
            output_dim=1,
            n_layers=args.num_layers,
        )
    elif args.model == "rnn":
        model = RNNpredictor(
            input_dim=len(test_dataset.x_var),
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

    # 获取初始参数
    test_dataset = FermentationData(
        work_dir=args.dataset, 
        train_mode=False, 
        y_var=["od_600"], 
        y_mean=y_mean, 
        y_std=y_std
    )
    x_mean, x_std = utils.get_norm_param(X=test_dataset.X_norm, x_cols=test_dataset.x_var)
    
    # 参数优化设置
    param_names = test_dataset.x_var
    initial_params = test_dataset.X[-1][-1] * x_std + x_mean
    epsilon = 1e-6

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

    # 主监控循环
    last_mod_time = os.path.getmtime(data_file_path)
    print(f"\n开始监控数据文件: {data_file_path}")
    print("按 Ctrl+C 停止监控")

    try:
        while True:
            try:
                current_mod_time = os.path.getmtime(data_file_path)
                if current_mod_time > last_mod_time:
                    print(f"\n检测到数据更新: {datetime.fromtimestamp(current_mod_time)}")
                    last_mod_time = current_mod_time

                    # 尝试读取数据文件
                    data_loaded = False
                    while not data_loaded:
                        try:
                            with FileLock(data_file_path + ".lock", timeout=1):
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

                    # 设置DataLoader
                    test_loader = torch.utils.data.DataLoader(
                        dataset=test_dataset, batch_size=1, num_workers=0, shuffle=False
                    )

                    # 参数优化
                    optimizer = ParameterOptimizer(
                        param_names, param_bounds, model, 
                        test_dataset, x_mean, x_std, y_mean, y_std
                    )
                    optimized_params = optimizer.optimize(initial_params, 0)

                    # 获取最佳预测结果
                    optimized_params_norm = _normalize_individual(optimized_params)
                    test_dataset.X[-1][-1] = optimized_params_norm
                    err, preds, labels = test(0, model, test_loader)
                    preds_denorm = preds * y_std + y_mean
                    labels_denorm = labels * y_std + y_mean
                    current_od600 = preds_denorm[-1]

                    # 计算指标
                    mse = np.square(np.subtract(preds_denorm, labels_denorm)).mean()
                    rmse = math.sqrt(mse)
                    refy = abs(preds_denorm[-1] - labels_denorm[-1]) / labels_denorm[-1] * 100

                    # 保存结果
                    result_saved = False
                    while not result_saved:
                        try:
                            with FileLock(result_file + ".lock", timeout=1):
                                # 读取现有数据
                                if os.path.exists(result_file):
                                    df = pd.read_excel(result_file)
                                else:
                                    df = pd.DataFrame(columns=result_columns)

                                # 添加新结果
                                new_row = {
                                    "Timestamp": datetime.now(),
                                    "Optimized_OD600": current_od600,
                                    "Best_Parameters": str(optimized_params.tolist()),
                                    "RMSE": rmse,
                                    "REFY": refy
                                }
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                
                                # 保存更新
                                df.to_excel(result_file, index=False)
                            
                            print(f"结果已保存至 {result_file}")
                            result_saved = True
                        except (Timeout, BlockingIOError):
                            print("结果文件被占用，1秒后重试...")
                            time.sleep(1)

                # 等待30秒再次检查
                time.sleep(30)

            except Exception as e:
                print(f"处理数据时发生错误: {str(e)}")
                time.sleep(30)  # 出错后等待30秒重试

    except KeyboardInterrupt:
        print("\n监控已停止")

	
	
	
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

