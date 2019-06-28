#---------回归模型--------------------
#---------导入数据--------------------

import numpy as np
from numpy import arange
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from pandas import set_option
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

filename='all_Fe.csv'
     #names=['h1','h2','h3','h4','t1','t2','t3','t4','t','abp1','abp2','abp3','abp4','abp5']
     #data=read_csv(filename,names=names)
data=read_csv(filename)

#-------------------------------------
#--------理解数据---------------------

print(data.shape)
print(data.dtypes)
set_option('display.width',130)         #显示栏宽度
print(data.head(10))                         #数据描述性统计
set_option('precision',2)                    #设置数据精度
print(data.describe())                       #查看描述性统计信息
set_option('precision',3) 
print(data.corr(method='pearson'))           #查看数据的皮尔逊相关系数

#-----------------------------------
#----------分离评估数据集------------


array=data.values
X=array[:,0:9]          #注意：包左不包右
Y=array[:,9:14]
validation_size=0.2
seed=7
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)

#---------------------------------------
#----------评估算法----------------------

num_folds=10   #10折交叉验证
#scoring='neg_mean_squared_error'    #均方误差来比较算法的准确度
scoring='neg_mean_absolute_error'    #平均绝对误差

models={}
models['LR']=LinearRegression()
models['LASSO']=Lasso()
models['EN']=ElasticNet()
models['KNN']=KNeighborsRegressor()
models['CART']=DecisionTreeRegressor()
#models['SVM']=SVR()


results=[]
for key in models:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s:%f(%f)'% (key,cv_result.mean(),cv_result.std()))

#-----------正态化数据-------------------

pipelines={}
pipelines['ScalerLR']=Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
pipelines['ScalerLASSO']=Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])
pipelines['ScalerEN']=Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])
pipelines['ScalerKNN']=Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
pipelines['ScalerCART']=Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])

print('                    ')
print('--------------------')
print('                    ')
results=[]
for key in pipelines:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_result=cross_val_score(pipelines[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s:%f(%f)'% (key,cv_result.mean(),cv_result.std()))
    
    
#------------调参改善算法-----------------
    
print('                    ')
print('--------------------')
print('                    ')

scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
model=KNeighborsRegressor()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result=grid.fit(X=rescaledX,y=Y_train)

print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))
cv_results=zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])
for mean,std,param in cv_results:
    print('%f(%f) with %r'%(mean,std,param))



print('                    ')
print('--------------------')
print('                    ')

scaler=StandardScaler().fit(X_train)
rescaledX=scaler.transform(X_train)
param_grid={'n_neighbors':[1,2,3,4,5,6,7]}
model=KNeighborsRegressor()
kfold=KFold(n_splits=num_folds,random_state=seed)
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result=grid.fit(X=rescaledX,y=Y_train)

print('最优：%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))
cv_results=zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])
for mean,std,param in cv_results:
    print('%f(%f) with %r'%(mean,std,param))

print('                       ')
print('-----------------------')
print('                       ')
#------------集成算法-----------------
#-------------------------------------
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
ensembles={}
ensembles['ScaledAB']=Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
ensembles['ScaledAB-KNN']=Pipeline([('Scaler',StandardScaler()),('ABKNN',AdaBoostRegressor
         (base_estimator=KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScaledAB-CART']=Pipeline([('Scaler',StandardScaler()),
                                     ('CART',AdaBoostRegressor(DecisionTreeRegressor()))])
ensembles['ScaledRFR']=Pipeline([('Scaler',StandardScaler()),
                                 ('RFR',RandomForestRegressor())])
ensembles['ScaledETR']=Pipeline([('Scaler',StandardScaler()),
                                 ('ETR',ExtraTreesRegressor())])
ensembles['ScaledGBR']=Pipeline([('Scaler',StandardScaler()),
                                 ('RBR',GradientBoostingRegressor())])

results=[]
for key in ensembles:
    kfold=KFold(n_splits=num_folds,random_state=seed)
    cv_result=cross_val_score(ensembles[key],X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_result)
    print('%s:%f(%f)'%(key,cv_result.mean(),cv_result.std()))
"""
# 用"""进行多行注释





