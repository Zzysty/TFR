"""
    svr模型
    搭建，训练以及评估
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from utils import compute_metrics, load_data, normalize_features_and_labels, sequence_decomposition, dimension_reduction
decompose = False  # 是否进行序列分解
# 设置随机种子
np.random.seed(1024)
# ------------------------------------------------ 数据准备 --------------------------------------------------------------
data, _ = load_data("../dataset/sh_carbon.csv")  # 读取数据
X_normalized, y_normalized, scaler_X, scaler_y = normalize_features_and_labels(data, "close")  # 数据标准化
# 移除特征序列中的第一个观测值以匹配差分后的标签序列长度
# X_normalized = X_normalized[1:, :]
# features_window, labels_window = create_sliding_windows(X_normalized, y_normalized, 30, 1)  # 滑窗处理

# train_data, val_data, test_data = split_data(features_window, labels_window)  # 数据划分

if decompose:
    # 序列分解
    IMFs, IMFs_combined = sequence_decomposition(y_normalized, "CEEMDAN")

    combined_features = np.hstack([X_normalized, IMFs_combined])  # 合并特征与IMFs
    print(f"IMFs合并特征-形状：{combined_features.shape}")

    # 降维
    reduced_features = dimension_reduction(combined_features)
    input_size = reduced_features.shape[1]  # 根据降维后的数据形状确定输入大小
else:
    reduced_features = X_normalized
    input_size = reduced_features.shape[1]

# 选择分割点（例如，使用前 80% 的数据作为训练集）
split_point = int(len(reduced_features) * 0.8)

# 创建训练集和测试集
X_train, y_train = reduced_features[:split_point], y_normalized[:split_point]
X_test, y_test = reduced_features[split_point:], y_normalized[split_point:]

# 确保 y_train 和 y_test 是一维数组
y_train = y_train.ravel()
y_test = y_test.ravel()

svr_model = SVR(kernel='rbf', C=1, epsilon=0.1, gamma='scale')
svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)

# 反归一化
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()  # 反归一化预测结果
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()  # 反归一化测试标签

predictions_df = pd.DataFrame({
    'Predictions': y_pred_original,
    'True_Values': y_test_original
})
# print(predictions_df)
# 保存到CSV
# predictions_df.to_csv('../vis/forecasting_results/svr.csv', index=False)

mse, mae, rmse, mape, r2, evs, me = compute_metrics(y_test_original, y_pred_original)
print(f"评估指标 --> MSE -> {mse:.4f}\n"
      f"-----------MAE -> {mae:.4f}\n"
      f"----------RMSE -> {rmse:.4f}\n"
      f"----------MAPE -> {mape:.4f}\n"
      f"------------R2 -> {r2:.4f}\n"
      f"-----------EVS -> {evs:.4f}\n"
      f"------------ME -> {me:.4f}")
