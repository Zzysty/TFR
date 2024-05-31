import random

import numpy as np
import pandas as pd
import torch
from PyEMD import CEEMDAN, EMD, EEMD, Visualisation
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score, max_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def seed_torch(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device():
    """设置设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device --> [{torch.cuda.get_device_name(device)}]")
    print('#---------------------------------------------------------------------------------------------------------#')
    return device


def load_data(filepath):
    """加载数据"""
    data = pd.read_csv(filepath, encoding='GBK', parse_dates=['date'], index_col='date').dropna()
    total_size = len(data)
    test_size = int(total_size * 0.1)  # 测试集占总数据的比例
    test_times = data.index[-test_size:]
    print(f"Data from --> {filepath}，测试集大小：{len(test_times)}，测试集时间：{test_times[0]} - {test_times[-1]}")
    print('#---------------------------------------------------------------------------------------------------------#')
    return data, test_times


def normalize_features_and_labels(data, label):
    """
        选取特征和标签，标准化
    """
    scaler_data = StandardScaler()
    scaler_labels = StandardScaler()

    # 选择特征和标签
    data_selected = data.drop(label, axis=1).values
    labels = data[label].values

    data_normalized = scaler_data.fit_transform(data_selected)
    labels_normalized = scaler_labels.fit_transform(labels.reshape(-1, 1))

    return data_normalized, labels_normalized, scaler_data, scaler_labels


def create_sliding_windows(data, labels, window_size=1, step_size=1):
    """
        创建滑动窗口数据集。
    """
    X, Y = [], []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:(i + window_size)])
        Y.append(labels[i + window_size - 1])
    return np.array(X), np.array(Y)


def split_data(features, labels, train_size=0.8, val_size=0.1, test_size=0.1):
    """
        划分数据集为训练集、验证集和测试集
    """
    total_samples = len(features)
    train_end = int(total_samples * train_size)
    val_end = train_end + int(total_samples * val_size)

    X_train = features[:train_end]
    Y_train = labels[:train_end]
    X_val = features[train_end:val_end]
    Y_val = labels[train_end:val_end]
    X_test = features[val_end:]
    Y_test = labels[val_end:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def create_dataloaders(train_set, validate_set, test_set, batch_size=32):
    """
        封装为Dataloader
    """
    train_dataset = TimeseriesDataset(*train_set)
    validate_dataset = TimeseriesDataset(*validate_set)
    test_dataset = TimeseriesDataset(*test_set)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validate_loader, test_loader


def inverse_standardize_data(scaled_data, scaler):
    """逆标准化数据"""
    return scaler.inverse_transform(scaled_data).flatten()


def compute_metrics(labels, predictions, eps=1e-3):
    """计算评估指标."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    mse = mean_squared_error(labels, predictions)  # 均方误差
    rmse = np.sqrt(mean_squared_error(labels, predictions))  # 均方根误差
    mae = mean_absolute_error(labels, predictions)  # 平均绝对误差
    r2 = r2_score(labels, predictions)  # R2
    mape = mean_absolute_percentage_error(labels, predictions)
    smape = 2.0 * np.mean(np.abs(predictions - labels) / (np.abs(predictions) + np.abs(labels))) * 100  # 对称平均绝对百分比误差
    evs = explained_variance_score(labels, predictions)
    me = max_error(labels, predictions)

    return mse, mae, rmse, mape, r2, evs, me


def sequence_decomposition(label, method='CEEMDAN'):
    """序列分解，EMD、EEMD、CEEMDAN方法"""
    decomposers = {'EMD': EMD, 'EEMD': EEMD, 'CEEMDAN': CEEMDAN}
    if method not in decomposers:
        raise ValueError(
            f"Unsupported decomposition method: {method}. Supported methods are: {list(decomposers.keys())}")

    decomposer = decomposers[method]()
    decomposer.noise_seed(1024)  # 设置随机种子
    IMFs = decomposer(label.flatten())

    print(f"Method:{method}, IMFs-Shape：{IMFs.shape}")
    IMFs_combined = np.stack(IMFs, axis=1)
    return IMFs, IMFs_combined


def plot_imfs(imfs, original_data, t=None, include_residue=False):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.labelsize'] = 20  # x轴刻度的字体大小
    plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
    plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小
    num_imfs = imfs.shape[0]
    t = t if t is not None else np.arange(imfs.shape[1])

    plt.figure(figsize=(10, 2 * (num_imfs + 1)))  # 调整figsize以适应额外的原始数据行
    # 首先绘制原始数据
    ax = plt.subplot(num_imfs + 1, 1, 1)
    ax.plot(t, original_data, label='Original Data', color='#1B3C73')  # 原始数据用蓝色
    ax.set_ylabel('Original data', fontdict={'fontfamily': 'Times New Roman', 'fontsize': 22})

    # 绘制各个IMFs
    for i in range(num_imfs - 1):  # 修改范围，为残差保留位置
        ax = plt.subplot(num_imfs + 1, 1, i + 2)
        ax.plot(t, imfs[i], label=f'IMF {i + 1}', color='#FF407D')  # IMF用红色
        ax.set_ylabel(f'IMF{i}', fontdict={'fontfamily': 'Times New Roman', 'fontsize': 22})

    # 特别处理残差部分
    if include_residue:
        ax = plt.subplot(num_imfs + 1, 1, num_imfs + 1)
        ax.plot(t, imfs[-1], label='Residue', color='#FF407D')  # 残差用绿色
        ax.set_ylabel('Residue', fontdict={'fontfamily': 'Times New Roman', 'fontsize': 22})
        ax.set_xlabel('Time', fontdict={'fontfamily': 'Times New Roman', 'fontsize': 22})

    plt.tight_layout()
    plt.savefig('./vis/IMFs.pdf', bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()


def dimension_reduction(combined_features):
    """降维"""
    pca = PCA(n_components=10)
    reduced_features = pca.fit_transform(combined_features)
    print(f"PCA降维-形状：{reduced_features.shape}")
    return reduced_features


class TimeseriesDataset(Dataset):
    """Dataset类，用于加载时间序列数据"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 使用.float()确保返回的张量为Float类型
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label
