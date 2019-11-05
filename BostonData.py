import pandas as pd  # 数据分析处理工具
import numpy as np # 快速操作结构数组的工具
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
df1 = pd.read_csv('boston.csv')#有表头的时候
names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
columns=['LSTAT','RM']###columns=['LSTAT','NOX','DIS']
df1.drop(columns,axis=1,inplace=True)###删除第12列数据df1.columns[[4，7，12]]lable=['MEDV']，必须加上axis=1表示删除的不是一行
features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS',
'RAD', 'TAX', 'PTRATIO', 'B']
X = df1[features]
X=np.array(X)
print(X.shape)
Y = df1['MEDV']
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

