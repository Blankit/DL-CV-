import time
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

'''
 1 读取数据，取列名
'''
df = pd.read_csv('项目_05葡萄酒分类问题资料/wine_data.csv',header=None)# header = None ，读取无列名的表
df.columns = ['f'+str(i) for i in range(len(df.columns))]
print(len(df))
# df.head()

'''
2. 选出特征部分
'''
col = df.columns[1:]
df_x = df[col]
# df_x.head()

'''
3. 选出标签部分
'''
df_y = df['f0']
# df_y

'''
4. 查看特征的统计信息及相关处理
'''
print(df_x.describe())
df_x.plot()

'''
发现
不同特征值的平均值中，最小为0.36,最大为746.89.数据存在严重倾斜

对数据做标准化处理。
将数据规范化到-1~1之间
'''

'''
标准化处理
'''
df_x = df_x.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
df_x.head()

# 显示标准化后的分布

fig, axis = plt.subplots(4,4,figsize=(12,10))# 设置子图是4*4，图大小的12*10

for i in range(4):
    for j in range(4):
        num = i*4+j+1
        axis[i][j].hist(df_x['f'+str(num)])# 画出f1~f13
        axis[i][j].set_title('f%d' % num)# 设置子图的标题
        if num>=13:
            break
plt.subplots_adjust(wspace=0.3,hspace=0.3)
# wspace,hspace：用于控制宽度和高度的百分比，比如subplot之间的间距

#规范化
x_norm = np.linalg.norm(df_x,axis=1)
x_norm = df_x.apply(lambda x: x/np.linalg.norm(x))
x_norm.head()

#显示规范化后的分布
fig, axis = plt.subplots(4,4,figsize=(12,10))# 设置子图是4*4，图大小的12*10

for i in range(4):
    for j in range(4):
        num = i*4+j+1
        axis[i][j].hist(x_norm['f'+str(num)])# 画出f1~f13
        axis[i][j].set_title('f%d' % num)# 设置子图的标题
        if num>=13:
            break
plt.subplots_adjust(wspace=0.3,hspace=0.3)
# wspace,hspace：用于控制宽度和高度的百分比，比如subplot之间的间距
'''
5. 数据切分
'''
x_train,x_test,y_train,y_test = train_test_split(x_norm, df_y,test_size =0.3)

'''
6. 构建模型
'''
# 构建模型，2个隐藏层，第1隐藏层50个神经元，第2隐藏层20个神经元，训练500周期
mlp = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=300)
mlp.fit(x_train, y_train)

'''
7. 模型预测
'''