import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# 读取数据，假设时间列名为'DATE'
file_path = './forecasting_results/all_model_predictions.csv'  # 替换为实际路径
data = pd.read_csv(file_path, encoding='GBK', parse_dates=['DATE'], index_col='DATE').dropna()
print(data.head())
# 平滑数据，例如使用7天的滚动平均
smoothed_data = data.rolling(window=15).mean()

# 指定颜色列表，确保颜色数量与模型数量匹配
colors = ['#1B3C73', '#4793AF', '#FFC470', '#DD5746', '#8B322C', '#FF407D']

# 绘制平滑后的数据
plt.figure(figsize=(14, 6))

# 设置全局字体和字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 20  # x轴刻度的字体大小
plt.rcParams['ytick.labelsize'] = 20  # y轴刻度的字体大小
plt.rcParams['legend.fontsize'] = 16  # 图例的字体大小

# 逐个模型绘制线条
for (column, color) in zip(smoothed_data.columns, colors):
    plt.plot(smoothed_data[column], label=column, color=color)

# 添加图例、标题和轴标签
plt.legend()
plt.xlabel('Time', fontdict={'family': 'Times New Roman', 'size': 22})
plt.ylabel('Carbon price', fontdict={'family': 'Times New Roman', 'size': 22})

# 设置坐标轴的标尺朝内
# plt.gca().tick_params(direction='in')
plt.tick_params(axis='both', direction='in')

# 控制x轴刻度显示个数
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(7))
# x 轴刻度旋转
# plt.xticks(rotation=10)
plt.savefig('./Comparison_model.pdf', format='pdf', bbox_inches='tight', dpi=600)  # 保存图表
# 显示图表
plt.show()
