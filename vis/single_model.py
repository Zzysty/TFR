# -*- coding: gbk -*-
# 引入所需库
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 给定的数据
data = {
    "Model": ["SVR", "LSTM", "GRU", "TCN", "TFR(AE+TCN)"],
    # "MSE": [9.3571, 2.3259, 2.1919, 2.1238, 0.7408],
    "MAE": [2.5391, 1.1829, 1.1153, 1.1655, 0.5828],
    "RMSE": [3.0589, 1.5251, 1.4805, 1.4573, 0.8607],
    "MAPE": [0.0607, 0.03, 0.0278, 0.0288, 0.0144]
}

# 将数据转换为长格式，适合用于Seaborn的绘图函数
df_long = pd.melt(pd.DataFrame(data), id_vars='Model', var_name='Metric', value_name='Error')

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 20  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小

# 绘制柱状图
plt.figure(figsize=(14, 6))
barplot = sns.barplot(x='Model', y='Error', hue='Metric', data=df_long,
                      palette=["#4793AF", "#FFC470", "#DD5746", "#8B322C"], edgecolor="black", linewidth=1)

# 增加图例
plt.legend()

# 设置图形标题及坐标轴标签
plt.title('')
plt.xlabel('')
plt.ylabel('Error', fontdict={'family': 'Times New Roman', 'size': 22})
plt.tick_params(axis='both', direction='in')

plt.savefig('single_model.pdf', format='pdf', bbox_inches='tight', dpi=600)
# 显示图形
plt.show()

