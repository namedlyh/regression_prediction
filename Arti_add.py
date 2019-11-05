import pandas as pd  # 数据分析处理工具
import numpy as np # 快速操作结构数组的工具
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
df1 = pd.read_csv(r'Arti_addP.txt',sep= ' ')
df1.columns =['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9','x10','x11', 'x12', 'x13', 'x14', 'x15', 'x16','x17', 'x18','x19','x20','y']
df1.to_csv('Arti_addH.txt', sep=' ',index=False)
features =['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8','x9','x10','x11', 'x12', 'x13', 'x14', 'x15', 'x16','x17', 'x18','x19','x20']
lable = ['y']
df1 = pd.read_csv('Arti_addH.txt',sep= ' ')#读的时候必须加,sep= ' '
X = df1[features]
X=np.array(X)
Y = df1[lable]
Y=np.array(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
# 数据标准化
ss_x = preprocessing.StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

# ss_y = preprocessing.StandardScaler()
# y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
# y_test = ss_y.transform(y_test.reshape(-1, 1))
#以下将数据转换成pands格式，方便进行数据分析
x_train=pd.DataFrame(
    x_train,columns=features)#将array转化为pandas
x_test=pd.DataFrame(
    x_test,columns=features)
# rf = RandomForestRegressor(max_features=5/13)#默认n_estimators=100
# x_train=x_train[['CRIM','ZN','CHAS']]
# x_train = np.array(x_train) #将pandas转化为 array
# rf.fit(x_train, y_train)#随机森林拟合
X1=ss_x.fit_transform(X)
X1=pd.DataFrame(
    X1,columns=features)#将array转化为pandas

