# -*- coding: gbk -*-
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 18  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小

# 准备数据
df = pd.DataFrame({
    'Model': ['(EMD)-SVR', '(EMD)-LSTM', '(EMD)-GRU', '(EMD)-TCN', '(EMD)-TFR*',
              '(EEMD)-SVR', '(EEMD)-LSTM', '(EEMD)-GRU', '(EEMD)-TCN', '(EEMD)-TFR*',
              '(CEEMDAN)-SVR', '(CEEMDAN)-LSTM', '(CEEMDAN)-GRU', '(CEEMDAN)-TCN', 'TFR'],
    'Metric': ['R2']*15,
    'Value': [
        0.3000, 0.6072, 0.5915, 0.7607, 0.9207,
        0.2268, 0.6352, 0.6040, 0.7501, 0.9264,
        0.3375, 0.7597, 0.6744, 0.7807, 0.9507
    ]
})

# 绘制条形图
plt.figure(figsize=(14, 8))
barplot = sns.barplot(x="Model", y="Value", hue="Metric", data=df, palette=["#FF407D"], edgecolor="black", linewidth=1)

# 计算趋势线
# 将模型名称转换为数值索引，以便进行数学计算
x_numeric = np.arange(len(df['Model']))
slope, intercept = np.polyfit(x_numeric, df['Value'], 1)  # 计算趋势线的斜率和截距
trend_line = slope * x_numeric + intercept  # 计算趋势线的值

# 绘制趋势线
plt.plot(df['Model'], trend_line, label='Trend Line', color='#1E1F22', linestyle='--')

# 去掉图例标题
plt.legend()

# 设置标题和轴标签
plt.title('')
plt.xlabel('')
plt.ylabel('R2', fontdict={'family': 'Times New Roman', 'size': 22})

# 旋转X轴标签以便它们更容易阅读
x_ticks = barplot.get_xticks()
barplot.xaxis.set_major_locator(FixedLocator(x_ticks))
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=45, horizontalalignment='right')

# 去掉网格线
plt.grid(False)

# 展示图形
plt.tight_layout()
plt.savefig('R2.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.show()


