import argparse
import os

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_model
from models.model import TCN, LSTM, GRU, TCN_Autoencoder
from train import run_train_model
from utils import seed_torch, set_device, load_data, normalize_features_and_labels, create_sliding_windows, split_data, \
    create_dataloaders, sequence_decomposition, dimension_reduction, plot_imfs


def main(args):
    # -------------------------------------------- 1. seed -------------------------------------------------------------
    seed_torch(args.seed)

    # -------------------------------------------- 2. device -----------------------------------------------------------
    device = set_device()

    # -------------------------------------------- 3. 数据 --------------------------------------------------------------
    data, test_times = load_data(args.data_path)  # 读取数据，测试集时间轴

    # 对数据和标签进行标准化
    data_normalized, labels_normalized, scaler_data, scaler_labels = normalize_features_and_labels(data, args.target)
    print(f"特征标准化-形状：{data_normalized.shape}，标签标准化-形状：{labels_normalized.shape}")

    if args.decompose:
        # 序列分解  EMD、EEMD、CEEMDAN 调用函数并请求可视化
        IMFs, IMFs_combined = sequence_decomposition(labels_normalized, method='CEEMDAN')
        plot_imfs(IMFs, labels_normalized, t=data.index, include_residue=True)

        combined_features = np.hstack([data_normalized, IMFs_combined])  # 合并特征与IMFs    该数据直接进 TAE
        print(f"IMFs合并特征-形状：{combined_features.shape}")  # (1960, 47)

        # 降维
        reduced_features = dimension_reduction(combined_features)
        input_size = reduced_features.shape[1]  # 根据降维后的数据形状确定输入大小
    else:
        # reduced_features = dimension_reduction(data_normalized)  # PCA
        reduced_features = data_normalized
        input_size = reduced_features.shape[1]

    # 滑窗处理  reduced_features  combined_features
    # TODO  TAE用combined_features，其他用reduced_features
    features_window, labels_window = create_sliding_windows(reduced_features, labels_normalized, args.seq_len,
                                                            args.step)

    # 划分数据集
    train_data, val_data, test_data = split_data(features_window, labels_window)

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(train_data, val_data, test_data, args.batch_size)

    # -------------------------------------------- 4. 模型 --------------------------------------------------------------
    if args.model == 'tcn':
        model = TCN(input_size, args.tcn_num_channels, args.tcn_kernel_size, args.output_size,
                    args.dropout).to(device)
    elif args.model == 'lstm':
        model = LSTM(input_size, args.lstm_hidden_dim, args.num_lstm_layers, args.output_size,
                     args.dropout).to(device)
    elif args.model == 'gru':
        model = GRU(input_size, args.gru_hidden_dim, args.num_gru_layers, args.output_size,
                    args.dropout).to(device)
    elif args.model == 'tae':
        # input_dim = args.input_size + IMFs_combined.shape[1]  # 输入维度 原数据 38 + 分解后的IMFs
        input_dim = args.input_size  # 输入维度
        model = TCN_Autoencoder(input_dim, args.encoded_space, args.tcn_num_channels, args.tcn_kernel_size, args.output_size,
                                args.dropout).to(device)

    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # -------------------------------------------- 5. train ------------------------------------------------------------
    print(f"开始训练模型 --> {args.model}")
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化器

    model_weights_path = os.path.join(args.model_path, args.weight_name)  # 模型权重路径 路径/文件名

    if os.path.isfile(model_weights_path):  # 如果存在预训练权重，则加载
        print(f"加载现有的权重从 --> {model_weights_path}")
        print('#---------------------------------------------------------------------------------------------------#')
        model.load_state_dict(torch.load(model_weights_path))
    else:
        print("未找到预训练权重，开始训练模型")
        print('#---------------------------------------------------------------------------------------------------#')
        run_train_model(model, train_loader, val_loader, criterion, optimizer, device, args.num_epochs,
                        model_weights_path)

    # -------------------------------------------- 6. evaluate ---------------------------------------------------------
    # 在评估时传递完整数据集的真实值和时间索引
    labels, predictions = evaluate_model(model, test_loader, device, scaler_labels)

    # 保存预测结果到CSV
    predictions_df = pd.DataFrame({
        'Predictions': predictions,
        'True_Values': labels
    })
    # 保存到CSV
    # predictions_df.to_csv('./vis/forecasting_results/tcn.csv', index=False)    # TODO


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CEEMDAN-TAE')

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    # --------------------------------------------  dataset settings ---------------------------------------------------
    parser.add_argument('--data_path', type=str, default='./dataset/sh_carbon.csv', help='which data to use')
    parser.add_argument('--seq_len', type=int, default=30, help='窗口大小')
    parser.add_argument('--step', type=int, default=1, help='步长')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--target', type=str, default='close', help='标签列')
    parser.add_argument('--decompose', type=bool, default=True, help='是否进行序列分解')

    # --------------------------------------------  models settings ----------------------------------------------------
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='LSTM层隐藏层维度')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='LSTM层数量')

    parser.add_argument('--gru_hidden_dim', type=int, default=64, help='GRU层隐藏层维度')
    parser.add_argument('--num_gru_layers', type=int, default=2, help='GRU层数量')

    parser.add_argument('--tcn_num_channels', type=int, default=[64], help='TCN层通道数量')  # TODO
    parser.add_argument('--tcn_kernel_size', type=int, default=3, help='TCN层卷积核大小')

    parser.add_argument('--encoded_space', type=int, default=8, help='自编码器编码空间维度')

    # --------------------------------------------  train settings -----------------------------------------------------
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--input_size', type=int, default=38, help='输入维度')
    parser.add_argument('--output_size', type=int, default=1, help='输出维度')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--dropout', type=int, default=0.3, help='dropout rate')
    parser.add_argument('--model', type=str, default='tae', choices=['tcn', 'lstm', 'gru', 'tae'])  # TODO
    parser.add_argument('--model_path', type=str, default='./checkpoints/tae/', help='保存权重文件的路径')  # TODO
    parser.add_argument('--weight_name', type=str, default=f'sh-h[64]-k3-lr0.001-d0.3-ep200.pth', help='权重文件名')   # TODO

    # --------------------------------------------  evaluate settings --------------------------------------------------
    parser.add_argument('--savefig_path', type=str, default='./vis/GT-sh.png', help='是否可视化预测结果')

    config = parser.parse_args()

    main(config)
