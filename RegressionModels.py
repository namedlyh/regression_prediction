
####只要数据集的格式统一，可以直接调用，对比八种机器学习回归模型在一些回归数据集的预测MSE,MAE和R2
###八种机器学习回归模型：线性回归模型，套索回归，岭回归，决策树，支持向量机回归，KNN，随机森林回归，改进的随机森林回归
###预测误差求20次的平均值
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import Arti_add  ###具体的生成数据集的.py名称
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR #svr有三种核，linear,poly,rbf
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
pool =ThreadPool()
lr_mse1=[]
lr_mae1=[]
lr_R21=[]
rr_mse1=[]
rr_mae1=[]
rr_R21=[]
lasso_mse1=[]
lasso_mae1=[]
lasso_R21=[]
rf_mse1=[]
rf_mae1=[]
rf_R21=[]
svr_mse1=[]
svr_mae1=[]
svr_R21=[]
dt_mse1=[]
dt_mae1=[]
dt_R21=[]
knn_mse1=[]
knn_mae1=[]
knn_R21=[]
ga_rf_mse1=[]
ga_rf_mae1=[]
ga_rf_R21=[]
for i in range(0,10):
    lr = LinearRegression() #线性回归
    rr = Ridge() #岭回归
    lasso = Lasso() #套索回归
    rf = RandomForestRegressor()
    ga_rf= RandomForestRegressor(n_estimators=30,max_features=45,min_samples_leaf=3)#max_features通常的取值为:1,sqrt为最大特征数的平方根
    knn= KNeighborsRegressor()
    #log2所有特征数的Log2值，None等于所有特征数
    #max_features如果是整数，代表考虑的最大特征数；如果是浮点数，表示对（N*max_features）取整，N是特征数总个数。
    svr = SVR(kernel = 'rbf')
    dt=DecisionTreeRegressor()
    #训练数据和测试数据
    x_train=Arti_add.x_train[Arti_add.features]
    x_train = np.array(x_train) #将pandas转化为 array
    x_test=Arti_add.x_test[Arti_add.features]
    x_test= np.array(x_test)#将pandas转化为 array
    #模型拟合
    rf.fit(x_train, Arti_add.y_train)  # 随机森林拟合
    lr.fit(x_train, Arti_add.y_train)
    rr.fit(x_train, Arti_add.y_train)
    lasso.fit(x_train, Arti_add.y_train)
    svr.fit(x_train, Arti_add.y_train)
    dt.fit(x_train, Arti_add.y_train)
    knn.fit(x_train, Arti_add.y_train)

    # names=['CRIM', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX']
    # names=Arti_add.features
    # x_train1=Arti_add.x_train[names]
    # x_train1 = np.array(x_train1) #将pandas转化为 arra
    # x_test1=Arti_add.x_test[names]
    # x_test1= np.array(x_test1)#将pandas转化为 array
    ga_rf.fit(x_train, Arti_add.y_train)
    #模型预测
    y_rf_ = rf.predict(x_test)
    # y_rf_ = y_rf_.reshape((-1, 1))###转成1列，之前是tuple(测试样本个数，),后面是空的，而目标变量是1列，必须一直，否则在进行相减时出问题
    y_lr_ = lr.predict(x_test)
    y_rr_ = rr.predict(x_test)
    y_lasso_ = lasso.predict(x_test)
    y_lasso_ = y_lasso_.reshape((-1, 1))###转成1列
    y_svr_= svr.predict(x_test)
    y_svr_ = y_svr_.reshape((-1, 1))###转成1列
    y_dt_=dt.predict(x_test)
    y_dt_ = y_dt_.reshape((-1, 1))###转成1列
    y_ga_rf_=ga_rf.predict(x_test)
    # y_ga_rf_ = y_ga_rf_.reshape((-1, 1))###转成1列
    y_knn_=knn.predict(x_test)
    #计算模型的预测误差

    #标签数据未归一化

    lr_mse = mean_squared_error(Arti_add.y_test, y_lr_)
    lr_mae = mean_absolute_error(Arti_add.y_test, y_lr_)
    lr_R2 = r2_score(Arti_add.y_test, y_lr_)

    rr_mse = mean_squared_error(Arti_add.y_test, y_rr_)
    rr_mae = mean_absolute_error(Arti_add.y_test, y_rr_)
    rr_R2 = r2_score(Arti_add.y_test, y_rr_)

    lasso_mse = mean_squared_error(Arti_add.y_test, y_lasso_)
    lasso_mae = mean_absolute_error(Arti_add.y_test, y_lasso_)
    lasso_R2 = r2_score(Arti_add.y_test, y_lasso_)

    rf_mse = mean_squared_error(Arti_add.y_test, y_rf_)
    rf_mae = mean_absolute_error(Arti_add.y_test, y_rf_)
    rf_R2 = r2_score(Arti_add.y_test, y_rf_)

    svr_mse = mean_squared_error(Arti_add.y_test, y_svr_)
    svr_mae = mean_absolute_error(Arti_add.y_test, y_svr_)
    svr_R2 = r2_score(Arti_add.y_test, y_svr_)

    dt_mse = mean_squared_error(Arti_add.y_test, y_dt_)
    dt_mae = mean_absolute_error(Arti_add.y_test, y_dt_)
    dt_R2 = r2_score(Arti_add.y_test, y_dt_)

    knn_mse = mean_squared_error(Arti_add.y_test, y_knn_)
    knn_mae = mean_absolute_error(Arti_add.y_test, y_knn_)
    knn_R2 = r2_score(Arti_add.y_test, y_knn_)

    ga_rf_mse = mean_squared_error(Arti_add.y_test, y_ga_rf_)
    ga_rf_mae = mean_absolute_error(Arti_add.y_test, y_ga_rf_)
    ga_rf_R2 = r2_score(Arti_add.y_test, y_ga_rf_)

    lr_mse1.append(lr_mse)
    lr_mae1.append(lr_mae)
    lr_R21.append(lasso_R2)
    rr_mse1.append(rr_mse)
    rr_mae1.append(rr_mae)
    rr_R21.append(rr_R2)
    lasso_mse1.append(lasso_mse)
    lasso_mae1.append(lasso_mae)
    lasso_R21.append(lasso_R2)
    rf_mse1.append(rf_mse)
    rf_mae1.append(rf_mae)
    rf_R21.append(rf_R2)
    svr_mse1.append(svr_mse)
    svr_mae1.append(svr_mae)
    svr_R21.append(svr_R2)
    dt_mse1.append(dt_mse)
    dt_mae1.append(dt_mae)
    dt_R21.append(dt_R2)
    knn_mse1.append(knn_mse)
    knn_mae1.append(knn_mae)
    knn_R21.append(knn_R2)
    ga_rf_mse1.append(ga_rf_mse)
    ga_rf_mae1.append(ga_rf_mae)
    ga_rf_R21.append(ga_rf_R2)
#     list求平均值
def Get_Average(list):
   sum = 0
   for item in list:
      sum += item
   return sum/len(list)
lr_mse=Get_Average(lr_mse1)
lr_mae=Get_Average(lr_mae1)
lr_R2=Get_Average(lr_R21)
rr_mse=Get_Average(rr_mse1)
rr_mae=Get_Average(rr_mae1)
rr_R2=Get_Average(rr_R21)
lasso_mse=Get_Average(lasso_mse1)
lasso_mae=Get_Average(lasso_mae1)
lasso_R2=Get_Average(lasso_R21)
svr_mse=Get_Average(svr_mse1)
svr_mae=Get_Average(svr_mae1)
svr_R2=Get_Average(svr_R21)
rf_mse=Get_Average(rf_mse1)
rf_mae=Get_Average(rf_mae1)
rf_R2=Get_Average(rf_R21)
dt_mse=Get_Average(dt_mse1)
dt_mae=Get_Average(dt_mae1)
dt_R2=Get_Average(dt_R21)
knn_mse=Get_Average(knn_mse1)
knn_mae=Get_Average(knn_mae1)
knn_R2=Get_Average(knn_R21)
ga_rf_mse=Get_Average(ga_rf_mse1)
ga_rf_mae=Get_Average(ga_rf_mae1)
ga_rf_R2=Get_Average(ga_rf_R21)
# 最大值
lr_max_mse=max(lr_mse1)
lr_max_mae=max(lr_mae1)
lr_max_R2=max(lr_R21)
rr_max_mse=max(rr_mse1)
rr_max_mae=max(rr_mae1)
rr_max_R2=max(rr_R21)
lasso_max_mse=max(lasso_mse1)
lasso_max_mae=max(lasso_mae1)
lasso_max_R2=max(lasso_R21)
svr_max_mse=max(svr_mse1)
svr_max_mae=max(svr_mae1)
svr_max_R2=max(svr_R21)
rf_max_mse=max(rf_mse1)
rf_max_mae=max(rf_mae1)
rf_max_R2=max(rf_R21)
dt_max_mse=max(dt_mse1)
dt_max_mae=max(dt_mae1)
dt_max_R2=max(dt_R21)
knn_max_mse=max(knn_mse1)
knn_max_mae=max(knn_mae1)
knn_max_R2=max(knn_R21)
ga_rf_max_mse=max(ga_rf_mse1)
ga_rf_max_mae=max(ga_rf_mae1)
ga_rf_max_R2=max(ga_rf_R21)
# 最小值
lr_min_mse=min(lr_mse1)
lr_min_mae=min(lr_mae1)
lr_min_R2=min(lr_R21)
rr_min_mse=min(rr_mse1)
rr_min_mae=min(rr_mae1)
rr_min_R2=min(rr_R21)
lasso_min_mse=min(lasso_mse1)
lasso_min_mae=min(lasso_mae1)
lasso_min_R2=min(lasso_R21)
svr_min_mse=min(svr_mse1)
svr_min_mae=min(svr_mae1)
svr_min_R2=min(svr_R21)
rf_min_mse=min(rf_mse1)
rf_min_mae=min(rf_mae1)
rf_min_R2=min(rf_R21)
dt_min_mse=min(dt_mse1)
dt_min_mae=min(dt_mae1)
dt_min_R2=min(dt_R21)
knn_min_mse=min(knn_mse1)
knn_min_mae=min(knn_mae1)
knn_min_R2=min(knn_R21)
ga_rf_min_mse=min(ga_rf_mse1)
ga_rf_min_mae=min(ga_rf_mae1)
ga_rf_min_R2=min(ga_rf_R21)
# 将每个算法的平均误差指标对应放在一起，方便画图
MSE = []
MAE = []
R2 = []
###每个MSE中保存的数据是依次回归模型在同一个数据集上的MSE取值，同理，MAE和R2
MSE.append(lr_mse),MAE.append(lr_mae),R2.append(lr_R2)
MSE.append(rr_mse),MAE.append(rr_mae),R2.append(rr_R2)
MSE.append(lasso_mse),MAE.append(lasso_mae),R2.append(lasso_R2)
MSE.append(rf_mse),MAE.append(rf_mae),R2.append(rf_R2)
MSE.append(svr_mse),MAE.append(svr_mae),R2.append(svr_R2)
MSE.append(dt_mse),MAE.append(dt_mae),R2.append(dt_R2)
MSE.append(knn_mse),MAE.append(knn_mae),R2.append(knn_R2)
MSE.append(ga_rf_mse),MAE.append(ga_rf_mae),R2.append(ga_rf_R2)
# 最大MSE,MAE,R2
MSE_max = []
MAE_max = []
R2_max = []
MSE_max.append(lr_max_mse),MAE_max.append(lr_max_mae),R2_max.append(lr_max_R2)
MSE_max.append(rr_max_mse),MAE_max.append(rr_max_mae),R2_max.append(rr_max_R2)
MSE_max.append(lasso_max_mse),MAE_max.append(lasso_max_mae),R2_max.append(lasso_max_R2)
MSE_max.append(rf_max_mse),MAE_max.append(rf_max_mae),R2_max.append(rf_max_R2)
MSE_max.append(svr_max_mse),MAE_max.append(svr_max_mae),R2_max.append(svr_max_R2)
MSE_max.append(dt_max_mse),MAE_max.append(dt_max_mae),R2_max.append(dt_max_R2)
MSE_max.append(knn_max_mse),MAE_max.append(knn_max_mae),R2_max.append(knn_max_R2)
MSE_max.append(ga_rf_max_mse),MAE_max.append(ga_rf_max_mae),R2_max.append(ga_rf_max_R2)
# 最小MSE,MAE,R2
MSE_min = []
MAE_min = []
R2_min = []
MSE_min.append(lr_min_mse),MAE_min.append(lr_min_mae),R2_min.append(lr_min_R2)
MSE_min.append(rr_min_mse),MAE_min.append(rr_min_mae),R2_min.append(rr_min_R2)
MSE_min.append(lasso_min_mse),MAE_min.append(lasso_min_mae),R2_min.append(lasso_min_R2)
MSE_min.append(rf_min_mse),MAE_min.append(rf_min_mae),R2_min.append(rf_min_R2)
MSE_min.append(svr_min_mse),MAE_min.append(svr_min_mae),R2_min.append(svr_min_R2)
MSE_min.append(dt_min_mse),MAE_min.append(dt_min_mae),R2_min.append(dt_min_R2)
MSE_min.append(knn_min_mse),MAE_min.append(knn_min_mae),R2_min.append(knn_min_R2)
MSE_min.append(ga_rf_min_mse),MAE_min.append(ga_rf_min_mae),R2_min.append(ga_rf_min_R2)
###一行一行写入
import csv
with open("Arti_add_MSE-MAE-R2.csv", "w") as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["LR", "岭回归", "Lasso", "RF", "SVR", "DT", "KNN", "RF_aga"])
    # 写入多行用writerows
    writer.writerows([np.transpose([MSE_min]), np.transpose([MSE]), np.transpose([MSE_max]),np.transpose([MAE_min]), np.transpose([MAE]), np.transpose([MAE_max]),np.transpose([R2_min]), np.transpose([R2]), np.transpose([R2_max])])
    # writer.writerows([[MSE_min], [MSE], [MSE_max],[MAE_min], [MAE], [MAE_max],[R2_min], [R2], [R2_max]])
    
np.savetxt('MSE.csv',np.transpose([MSE]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('MSE_max.csv',np.transpose([MSE_max]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('MSE_min.csv',np.transpose([MSE_min]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('MAE.csv',np.transpose([MAE]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('MAE_max.csv',np.transpose([MAE_max]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('MAE_min.csv',np.transpose([MAE_min]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('R2.csv',np.transpose([R2]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('R2_max.csv',np.transpose([R2_max]),fmt='%.4f',delimiter=' ')#####将数据存储下来
np.savetxt('R2_min.csv',np.transpose([R2_min]),fmt='%.4f',delimiter=' ')#####将数据存储下来

print('mse of linear regression is: '+str(lr_mse)+',mae is: '+str(lr_mae)+',R2 is: '+str(lr_R2))
print('mse of ridge is: '+str(rr_mse)+ ',mae is: '+str(rr_mae)+',R2 is: '+str(rr_R2))
print('mse of lasso is: '+str(lasso_mse)+ ',mae is: '+str(lasso_mae)+',R2 is: '+str(lasso_R2))
print('mse of rf is: '+str(rf_mse)+ ',mse of rf is: '+ str(rf_mae)+',R2 of rf is: '+ str(rf_R2))
print('mse of svr is: '+str(svr_mse)+ ',mae is: '+str(svr_mae)+',R2 is: '+str(svr_R2))
print('mse of decison tree is: '+str(dt_mse)+ ',mae is: '+str(dt_mae)+',R2 is: '+str(dt_R2))
print('mse of knn is: '+str(knn_mse)+ ',mae is: '+str(knn_mae)+',R2 is: '+str(knn_R2))
print('mse of ga rf is: '+str(ga_rf_mse)+ ',mae is: '+str(ga_rf_mae)+',R2 is: '+str(ga_rf_R2))



# 画出预测值与真实值的差异
x_num=np.arange(1, Arti_add.y_test.shape[0] + 1, 1)
# x_num=np.arange(1, 100 + 1, 1)
plt.figure()

Line1=plt.plot(x_num, Arti_add.y_test, label=u'验证数据中标签值', color='b', linestyle='-', marker='>')
# Line2=plt.plot(x_num,y_lr_,label=u'LR预测值',color='k',linestyle='--',marker='v')
# Line3=plt.plot(x_num,y_rr_,label=u'岭回归预测值',color='r',linestyle='-.',marker='^')
# Line4=plt.plot(x_num,y_lasso_,label=u'Lasso预测值',color='y',linestyle=':',marker='<')
# Line5=plt.plot(x_num,y_rf_,label=u'RF预测值',color='g',linestyle='-',marker='>')
# Line5=plt.plot(x_num,y_rr_,label=u'DNN预测值',color='g',linestyle=':',marker='v')
# line6=plt.plot(x_num,y_svr_,label=u'SVR预测值',color='c',linestyle='--',marker='p')
# Line7=plt.plot(x_num,y_dt_,label='DT预测值',color='m',linestyle='-.',marker='*')
# Line8=plt.plot(x_num,y_knn_,label='knn预测值',color='chocolate',linestyle='-.',marker='*')
# Line9=plt.plot(y_ga_rf_,label=u'RF_aga预测值',color='r',linestyle=':',marker='+')
plt.legend(loc='best',fontsize='x-large')
plt.figure(1,figsize=(24, 16))
plt.xlabel("验证集样本序号",fontsize=15)
# plt.ylabel("error",fontsize=20)
plt.title("'舰艇探测导弹能力'验证集标签值和DNN预测值对比",fontsize=10)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.show()
###重复做一次
plt.figure()

Line1=plt.plot(x_num, Arti_add.y_test, label=u'真实值', color='b', linestyle='-', marker='>')
# Line2=plt.plot(x_num,y_lr_,label=u'LR预测值',color='k',linestyle='--',marker='v')
# Line3=plt.plot(x_num,y_rr_,label=u'岭回归预测值',color='r',linestyle='-.',marker='^')
# Line4=plt.plot(x_num,y_lasso_,label=u'Lasso预测值',color='y',linestyle=':',marker='<')
# Line5=plt.plot(x_num,y_rf_,label=u'RF预测值',color='g',linestyle='-',marker='>')
# Line5=plt.plot(x_num,y_rf_,label=u'预测值',color='g',linestyle='-',marker='>')
# line6=plt.plot(x_num,y_svr_,label=u'SVR预测值',color='c',linestyle='--',marker='p')
# Line7=plt.plot(x_num,y_dt_,label='DT预测值',color='m',linestyle='-.',marker='*')
# Line8=plt.plot(x_num,y_knn_,label='knn预测值',color='chocolate',linestyle='-.',marker='*')
Line9=plt.plot(x_num,y_ga_rf_,label=u'预测值',color='g',linestyle=':',marker='v')
plt.legend(loc='best',fontsize='x-large')
plt.figure(1,figsize=(24, 16))
plt.xlabel("验证集样本序号",fontsize=15)
# plt.ylabel("error",fontsize=20)
# plt.title("测试样本下的'兵力到位率'真实值和测试值对比",fontsize=10)
plt.title("'舰艇探测导弹能力'验证集标签值和DNN预测值对比",fontsize=10)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.show()
#
error1= Arti_add.y_test - y_lr_
error2= Arti_add.y_test - y_rr_
error3= Arti_add.y_test - y_lasso_
error4= Arti_add.y_test - y_rf_
error5= Arti_add.y_test - y_svr_
error6= Arti_add.y_test - y_dt_
error7= Arti_add.y_test - y_knn_
error8= Arti_add.y_test - y_ga_rf_


# 画出真实值与测试值之间的误差线
plt.figure()
# Line1=plt.plot(x_num,error1,linestyle='-',label=u'LR预测误差',color='k',marker='o')
# Line2=plt.plot(x_num,error2,linestyle='--',label=u'岭回归预测误差',color='r',marker='*')
# Line3=plt.plot(x_num,error3,linestyle='-.',label=u'Lasso预测误差',color='y',marker='s')
# Line4=plt.plot(x_num,error2,linestyle=':',label=u'标签值和预测值差值',color='k',marker='+')#
# Line4=plt.plot(x_num,error4,linestyle=':',label=u'RF预测误差',color='b',marker='>')#
# Line5=plt.plot(x_num,error5,linestyle='-',label=u'SVR预测误差',color='c',marker='<')
# Line6=plt.plot(x_num,error6,linestyle='--',label=u'DT预测误差',color='m',marker='v')
# Line7=plt.plot(x_num,error7,linestyle='-.',label=u'KNN预测误差',color='chocolate',marker='p')
# Line8=plt.plot(x_num,error8,linestyle=':',label=u'真实值和测试值差值',color='r',marker='+')#
plt.legend(loc='best',fontsize='x-large')
plt.figure(1,figsize=(24, 16))
plt.xlabel("验证集样本序号",fontsize=15)
# plt.ylabel("预测误差",fontsize=15)
plt.title("'舰艇探测导弹能力'验证集标签值和DNN预测值差值",fontsize=10)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.show()

plt.figure()
# Line1=plt.plot(x_num,error1,linestyle='-',label=u'LR预测误差',color='k',marker='o')
# Line2=plt.plot(x_num,error2,linestyle='--',label=u'岭回归预测误差',color='r',marker='*')
# Line3=plt.plot(x_num,error3,linestyle='-.',label=u'Lasso预测误差',color='y',marker='s')
# Line4=plt.plot(x_num,error4,linestyle=':',label=u'真实值和测试值差值',color='b',marker='>')#
# Line4=plt.plot(x_num,error4,linestyle=':',label=u'RF预测误差',color='b',marker='>')#
# Line5=plt.plot(x_num,error5,linestyle='-',label=u'SVR预测误差',color='c',marker='<')
# Line6=plt.plot(x_num,error6,linestyle='--',label=u'DT预测误差',color='m',marker='v')
# Line7=plt.plot(x_num,error7,linestyle='-.',label=u'KNN预测误差',color='chocolate',marker='p')
Line8=plt.plot(x_num,error8,linestyle=':',label=u'标签值和预测值差值',color='k',marker='+')#
plt.legend(loc='best',fontsize='x-large')
plt.figure(1,figsize=(24, 16))
plt.xlabel("验证集样本序号",fontsize=15)
# plt.ylabel("预测误差",fontsize=15)
plt.title("'舰艇探测导弹能力'验证集标签值和DNN预测值差值",fontsize=10)
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.show()

list1=['LR','岭回归','Lasso','RF','SVR','DT','KNN','RF_aga']
n=np.arange(len(MSE))#####算法的个数
# X=np.arange(MSE.shape[1])
width=0.3
fig,ax=plt.subplots(figsize=(24,16))
b1=ax.bar(n-width,MSE,width,label='MSE')
b2=ax.bar(n,MAE,width,label='MAE',tick_label=list1)
b3=ax.bar(n+width,R2,width,label='R2')
for b in b1+b2+b3:
    h=b.get_height()
    ax.text(b.get_x()+b.get_width()/2,h,'%.3f'%h,ha='center',va='bottom',fontsize=26)

plt.tick_params(labelsize=30)
plt.title("各回归算法MSE,MAE,R2",fontsize=40)
plt.xlabel("回归算法",fontsize=40)
plt.ylabel("平均预测误差",fontsize=40)
plt.legend(fontsize=40)
#以下两句是为了显示中文用
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()


# 最大预测误差
fig,ax=plt.subplots(figsize=(24,16))
b1=ax.bar(n-width,MSE_max,width,label='MSE')
b2=ax.bar(n,MAE_max,width,label='MAE',tick_label=list1)
b3=ax.bar(n+width,R2_max,width,label='R2')

for b in b1+b2+b3:
    h=b.get_height()
    ax.text(b.get_x()+b.get_width()/2,h,'%.3f'%h,ha='center',va='bottom',fontsize=26)

plt.tick_params(labelsize=30)
plt.title("各回归算法MSE,MAE,R2",fontsize=40)
plt.xlabel("回归算法",fontsize=40)
plt.ylabel("最大预测误差",fontsize=40)
plt.legend(fontsize=40)
#以下两句是为了显示中文用
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()

#最小预测误差
fig,ax=plt.subplots(figsize=(24,16))
b1=ax.bar(n-width,MSE_min,width,label='MSE')
b2=ax.bar(n,MAE_min,width,label='MAE',tick_label=list1)
b3=ax.bar(n+width,R2_min,width,label='R2')

for b in b1+b2+b3:
    h=b.get_height()
    ax.text(b.get_x()+b.get_width()/2,h,'%.3f'%h,ha='center',va='bottom',fontsize=26)

plt.tick_params(labelsize=30)
plt.title("各回归算法MSE,MAE,R2",fontsize=40)
plt.xlabel("回归算法",fontsize=40)
plt.ylabel("最小预测误差",fontsize=40)
plt.legend(fontsize=40)
#以下两句是为了显示中文用
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()

print('Score:', rf.score(x_test, Arti_add.y_test))

