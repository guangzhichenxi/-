import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler

# 配置参数
CONFIG = {
    "data_path": "./input_data/train.csv",
    "output_path": "./output_images/",
    "datetime_format": "%Y-%m-%d %H:%M:%S",
    "category_columns": ['season', 'holiday', 'workingday', 'weather', 
                         'weekday', 'year', 'month', 'hour'],
}

# 创建输出目录
os.makedirs(CONFIG["output_path"], exist_ok=True)

# Matplotlib 和 Seaborn 配置
params = {
    'legend.fontsize': 'large',
    'figure.figsize': (20, 10),
    'axes.labelsize': 'large',
    'axes.titlesize': 'large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
    'font.sans-serif': 'SimHei',  # 中文支持
    'axes.unicode_minus': False  # 防止负号显示问题
}
sn.set_style('whitegrid')
sn.set_context('talk')
plt.rcParams.update(params)

# 加载数据
if os.path.exists(CONFIG["data_path"]):
    train = pd.read_csv(CONFIG["data_path"])
else:
    raise FileNotFoundError(f"File not found: {CONFIG['data_path']}")

# 检查数据是否有缺失值
if train.isnull().any().any():
    print("Missing values detected! Filling missing values.")
    train.fillna(method='ffill', inplace=True)

# 特征分解
def split_datetime(data):
    data['date'] = pd.to_datetime(data['datetime']).dt.date
    data['weekday'] = pd.to_datetime(data['datetime']).dt.dayofweek

    data['year'] = pd.to_datetime(data['datetime']).dt.year - 2011
    data['month'] = pd.to_datetime(data['datetime']).dt.month
    data['hour'] = pd.to_datetime(data['datetime']).dt.hour
    return data

data_train = split_datetime(train)

# 类型转换
def type_convert(data, category_columns):
    for col in category_columns:
        data[col] = data[col].astype('category')
    return data

data_train = type_convert(data_train, CONFIG["category_columns"])

# 检查数据内容
print(data_train.info())

# 数据分布描述
print(data_train.describe())

# 绘图函数
def save_plot(fig, filename):
    fig.tight_layout()
    fig_path = os.path.join(CONFIG["output_path"], filename)
    plt.savefig(fig_path)
    print(f"Saved: {fig_path}")
    plt.close(fig)

# 小提琴图：年份分析
fig, ax = plt.subplots()
sn.violinplot(data=data_train, x='year', y='count', ax=ax)
ax.set(title='Analysis of Year', xlabel='Year', ylabel='Count')
save_plot(fig, "Analysis_of_year.jpg")

# 点图：小时与季节分析
fig, ax = plt.subplots()
sn.pointplot(data=data_train, x='hour', y='count', hue='season', ax=ax)
ax.set(title='Analysis of Season on Hour', xlabel='Hour', ylabel='Count')
save_plot(fig, "Analysis_of_season_on_hour.jpg")

# 柱状图：月份分析
fig, ax = plt.subplots()
sn.barplot(data=data_train, x='month', y='count', ax=ax)
ax.set(title='Analysis of Month', xlabel='Month', ylabel='Count')
save_plot(fig, "Analysis_of_month.jpg")

# 点图：星期与小时分析
fig, ax = plt.subplots()
sn.pointplot(data=data_train, x='hour', y='count', hue='weekday', ax=ax)
ax.set(title='Analysis of Weekday on Hour', xlabel='Hour', ylabel='Count')
save_plot(fig, "Analysis_of_weekday_on_hour.jpg")

# 假日与工作日分析
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
sn.barplot(data=data_train, x='holiday', y='count', hue='season', ax=ax1)
sn.barplot(data=data_train, x='workingday', y='count', hue='season', ax=ax2)
ax1.set(title='Holiday vs Count by Season', xlabel='Holiday', ylabel='Count')
ax2.set(title='Working Day vs Count by Season', xlabel='Working Day', ylabel='Count')
save_plot(fig, "Analysis_of_holiday_vs_workingday.jpg")

# 相关性热图
corrMatt = data_train[["temp", "atemp", "humidity", "windspeed", "casual", 
                       "registered", "count"]].corr()
mask = np.triu(np.ones_like(corrMatt, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 8))
sn.heatmap(corrMatt, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set(title='Correlation Analysis')
save_plot(fig, "Correlation_analysis.jpg")
