# KAGGLE_houseprice_predict
we predict average house price with several parameters
ranking standard is rmse of prediction

#python
#first step: data cleaning
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))


train_ID = train['ID']
test_ID = test['ID']

train.drop("ID", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


train_columns = train.loc[:,'Paved Drive':'1stFloorArea'].columns
test = test.loc[:,train_columns]
test.head()
all_data = pd.concat((train.loc[:,'Paved Drive':'1stFloorArea'],
                      test.loc[:,'Paved Drive':'1stFloorArea']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
print('nums of skewed feature %s' %len(skewed_feats))

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
y = train.SalePrice



#feature selction with PCA in SKLEARN
#introduction link:http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
fit = pca.fit(X_train)
print("Explained Variance: %s" % fit.explained_variance_ratio_)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#feature selection with matetrial feature importance (choose by rank)(take first 50 variables as example)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
x_train, x_test, y_train, y_test= train_test_split(X_train,y,test_size=0.2) 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
reg2 = RandomForestRegressor(n_estimators=200)
reg2.fit(x_train,y_train)
imp=reg2.feature_importances_
imp = pd.DataFrame({'feature': X_train.columns, 'score': imp})
print(imp.sort_values(['score'], ascending=[0]))  
imp = imp.sort_values(['score'], ascending=[0])
select_feature50=imp['feature'][:50]
xtrain_feature50=X_train.loc[:,select_feature50]
xtest_feature50=X_test.loc[:,select_feature50]

############################################################################################################## train the model
#regress with radomforest
import numpy as np
R=np.zeros((10,10))
n_range=range(1,6)
k_range=[180,200,220,250,270]
for N in n_range:
    b=0
    for K in k_range:
        pca = PCA(n_components=K)
        fit = pca.fit(all_data)
        alldata_pca = pca.transform(all_data)
        Xtrain_pca=alldata_pca[:train.shape[0]]
        x_train, x_test, y_train, y_test= train_test_split(Xtrain_pca,y,test_size=0.2) 
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
        reg2 = RandomForestRegressor(n_estimators =200)
        reg2.fit(x_train,y_train)
        Pred2=reg2.predict(x_test)
        mse2=mean_squared_error(y_test, Pred2)
        rmse2=sqrt(mse2)
        R[N][b]=rmse2
        b=b+1
        print(str(N)+' '+str(K))

#regress with SVC
from sklearn.svm import SVC
x_train, x_test, y_train, y_test= train_test_split(xtrain_feature100,y,test_size=0.05) 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05)
clf=SVC(kernel='rbf')
param_grid01 = {'C': [ 1], 'gamma':[0.001] }
svc_cv=GridSearchCV(estimator=clf, param_grid=param_grid01, cv= 10)

#tune parameter with Gridsearchcv
x_train, x_test, y_train, y_test= train_test_split(xtrain_feature150,y,test_size=0.2) 
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
reg = RandomForestRegressor(n_estimators=100)
from sklearn.grid_search import GridSearchCV
param_grid = { 
    'n_estimators': [200,500,800],
}
CV_rfc = GridSearchCV(estimator=reg, param_grid=param_grid, cv= 10)
CV_rfc.fit(x_train, y_train)
Pred4=CV_rfc.predict(x_test)
mse4=mean_squared_error(y_test, Pred4)
rmse4=sqrt(mse4)

#use the model which generate the smallest test data rmse

#############################################################################################################predict house price
y_pred1=reg.predict(Xtest_pca)
y_pred_final1 = np.expm1(y_pred1)
#save data to file
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = y_pred_final1
sub.to_csv('submission.csv',index=False)



