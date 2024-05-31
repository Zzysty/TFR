"""
    评估部分相关代码
"""
import numpy as np
import torch

from matplotlib import pyplot as plt

from utils import compute_metrics, inverse_standardize_data


def evaluate_model(model, data_loader, device, scaler_label):
    print('#---------------------------------------------------------------------------------------------------#')
    print('#--------------------------------------------开始评估模型----------------------------------------------#')
    model.eval()  # 将模型设置为评估模式
    predictions, labels_list = [], []

    with torch.no_grad():  # 在评估过程中不计算梯度
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features)
            predictions.extend(output.cpu().numpy())  # 将预测结果转移到CPU并转换为numpy数组
            labels_list.extend(labels.cpu().numpy())  # 将实际标签转移到CPU并转换为numpy数组

    # 逆标准化
    predictions = inverse_standardize_data(predictions, scaler_label)
    labels_list = inverse_standardize_data(labels_list, scaler_label)

    # 计算指标
    mse, mae, rmse, mape, r2, evs, me = compute_metrics(labels_list, predictions)
    print(f"评估指标 --> MSE -> {mse:.4f}\n"
          f"-----------MAE -> {mae:.4f}\n"
          f"----------RMSE -> {rmse:.4f}\n"
          f"----------MAPE -> {mape:.4f}\n"
          f"------------R2 -> {r2:.4f}\n"
          f"-----------EVS -> {evs:.4f}\n"
          f"------------ME -> {me:.4f}")

    # 可视化测试集的预测值和真实值
    plt.figure(figsize=(14, 7))
    plt.plot(predictions, label="Test Set Predictions", color='blue')
    plt.plot(labels_list, label="Test Set True Values", color='red', linestyle='dashed')
    plt.title("Test Set Predictions vs True Values")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return labels_list, predictions

