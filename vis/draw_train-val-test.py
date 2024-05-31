# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=['date'], index_col='date')
    total_size = len(df)
    test_size = int(total_size * 0.1)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size - test_size
    return df['close'], train_size, val_size, test_size


# Load data
df, train_size, val_size, test_size = load_data('../dataset/sh_carbon.csv')

plt.figure(figsize=(12, 6))

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 20  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小

# 绘制训练集部分
plt.plot(df.iloc[:train_size], label="Training set", color='#1E1F22')

# 绘制验证集部分
plt.plot(df.iloc[train_size:train_size + val_size], label="Validation set", color='#1B3C73')

# 绘制测试集部分
plt.plot(df.iloc[-test_size:], label="Test set", color='#FF407D')

plt.xlabel('Time', fontdict={'family': 'Times New Roman', 'size': 22})
plt.ylabel('Carbon Price', fontdict={'family': 'Times New Roman', 'size': 22})

# 设置 x 轴和 y 轴刻度朝内
plt.tick_params(axis='both', direction='in')
plt.legend(loc='lower right')
plt.tight_layout()
# 去掉网格
plt.grid(False)

# 如果需要保存图像，取消注释以下行并指定保存路径
plt.savefig('Shanghai.pdf', format='pdf', bbox_inches='tight', dpi=600)

plt.show()
