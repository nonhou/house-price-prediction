"""二手房价格预测：数据清洗 + 特征工程 + LinearSVR 回归 + 真实值/预测值对比图

注：本脚本依据实训报告中的代码片段整理。报告中部分中间变量为节选，
以下按报告的建模流程补全为可运行版本；模型与特征工程方法未作改动。
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score

# ---------- 1. 导入数据 ----------
data = pd.read_csv('data.csv')

# ---------- 2. 清洗数据 ----------
if 'Unnamed: 0' in data.columns:
    del data['Unnamed: 0']
data.dropna(axis=0, how='any', inplace=True)          # 删除空值
data['总价'] = data['总价'].map(lambda e: e.replace('万', ''))
data['总价'] = data['总价'].astype(float)
data['建筑面积'] = data['建筑面积'].map(lambda p: p.replace('平米', ''))
data['建筑面积'] = data['建筑面积'].astype(float)

# ---------- 3. 特征工程 ----------
copy_d = data.copy()
copy_d[['室', '厅', '卫']] = copy_d['户型'].str.extract(r'(\d+)室(\d+)厅(\d+)卫')
copy_d['室'] = copy_d['室'].astype(float)
copy_d['厅'] = copy_d['厅'].astype(float)
copy_d['卫'] = copy_d['卫'].astype(float)

new_data = copy_d[['总价', '建筑面积', '室', '厅', '卫']]
new_data.dropna(axis=0, how='any', inplace=True)

x = new_data.drop(['总价'], axis=1)
y = new_data['总价']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

# ---------- 4. 建模 ----------
svr = LinearSVR(max_iter=10000)
svr.fit(x_train, y_train)

# ---------- 5. 评估 ----------
y_pred = svr.predict(x_test)
print('R2 =', r2_score(y_test, y_pred))

# ---------- 6. 预测：建筑面积 180 平米、4 房 2 厅 2 卫 ----------
x_pred = pd.DataFrame({'建筑面积': [180.0], '室': [4.0], '厅': [2.0], '卫': [2.0]})
print('预测总价（万）=', svr.predict(x_pred)[0])

# ---------- 7. 可视化 ----------
y_p = pd.Series(y_pred)
y_t = pd.Series(y_test.values, index=y_p.index)

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(y_t, label='真实值', marker='o')
plt.plot(y_p, label='预测值', marker='x', linestyle='--')
plt.title('测试集真实值与预测值对比')
plt.xlabel('样本编号')
plt.ylabel('房子总价(万)')
plt.legend()
plt.show()
