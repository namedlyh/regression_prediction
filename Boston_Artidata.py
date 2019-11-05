import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
df = pd.read_csv('boston.csv')#有表头的时候
features=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
X = df[features]
X=np.array(X)
Y = df['MEDV']
lable=['MEDV']
Y=np.array(Y)
n1=X.shape[0]##行
m1=87####要增加的噪音变量个数
F_add=np.random.rand(n1, m1)
F=np.c_[X,F_add]
#####生成特征名称
prepare_list = []
for i in range(m1):
    j=i+1
    prepare_list.append('x' + str(j))
features = features + prepare_list ######list合并,所有的特征名称
names=features+lable######所有的变量名称，包含目标变量名称
###重新组成数据集文件
Boston_ArtiD = np.c_[F,Y]###包含特征数据和目标变量数据
np.savetxt('Boston_ArtiD.txt',Boston_ArtiD,fmt='%.4f',delimiter=' ')
#####读出来
df1 = pd.read_csv(r'Boston_ArtiD.txt',sep= ' ')
df1.columns =names
df1.to_csv('Boston_ArtiDH.txt', sep=' ',index=False)
df1 = pd.read_csv('Boston_ArtiDH.txt',sep= ' ')#读的时候必须加,sep= ' '
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