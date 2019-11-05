import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
df = pd.read_csv(r'communities.data',sep=',')
df1=np.array(df)
n1=df1.shape[0]
m1=df1.shape[1]
features = []
for i in range(m1-1):
    j=i+1
    features.append('x' + str(j))
lable=['y']
names=features+lable######所有的变量名称，包含目标变量名称
df.columns =names
df=pd.DataFrame(
    df1,columns=names)
###缺失值用均值填充
df=df.fillna(df.mean())####必须将值赋予df才真正改变了df
# print(df.isnull())###判断数据是否有缺失
###查看x4列标题下标称数据有多少个不同值，以便后面对数据进行离散化，将print数据输出到txt中
# f = open("out.txt", "w+")
# np.set_printoptions(threshold=np.inf)##为了让print数据显示全
# print('Different attribute number of x4 is: ',df['x4'].value_counts(),file=f) ##计算pandas或series中相同数据出现的频率
# f.close()
# 将标称型数据转换成具体的数值
le=preprocessing.LabelEncoder()
le=le.fit(df['x4'])
df['x4'] = le.transform(df['x4'])
X=np.array(df[features])
Y = np.array(df[lable])
#先进行训练集和测试集划分
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

###针对不同的模型，对数据集进行数据预处理---one-hot编码，数据归一化
###第一种情况，针对树模型，直接将标称数据赋予0，1这样的值

####第二种情况，将无大小表示的标称数据，进行one-hot编码,在分类情况较多时不适合

# 数据标准化
ss_x = preprocessing.StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
###将缺失值处理以及标称数据转换后的数据保存下来，并加上列标题
# np.savetxt('communitiesD.txt',df,fmt='%.4f',delimiter=' ')#
# #####读出来
# df1 = pd.read_csv(r'communitiesD.txt',sep= ' ')
# df1.columns =names
# df1.to_csv('communitiesDH.txt', sep=' ',index=False)

#以下将数据转换成pands格式，方便进行数据分析
x_train=pd.DataFrame(
    x_train,columns=features)#将array转化为pandas
x_test=pd.DataFrame(
    x_test,columns=features)