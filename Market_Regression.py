import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

#Read both Train and test data
train = pd.read_csv('/Train1.csv')

test = pd.read_csv('/Test1.csv')


train.head()

#Replacing the missing values in variable 'Item Visibility' with the mean value 
train['Item_Visibility'] = train['Item_Visibility'].replace(0,np.mean(train['Item_Visibility']))
test['Item_Visibility'] = test['Item_Visibility'].replace(0,np.mean(test['Item_Visibility']))

#Standardizing values in the variable 'Outlet_Establishment_Year'
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year']

#replacing missing values with the most frequent category 
train['Outlet_Size'].fillna('Small',inplace=True)
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

#Creating list to form dummy values for categorical variables
mylist = list(train.select_dtypes(include=['object']).columns)
mylist1 = list(test.select_dtypes(include=['object']).columns)


#delete Item Identifier as it is of no use for analysis
del(mylist[0])

#delete Item Identifier as it is of no use for analysis
del(mylist1[0])

#creating dummy variables
dummies = pd.get_dummies(train[mylist], prefix= mylist)
train.drop(mylist, axis=1, inplace = True)
X = pd.concat([train,dummies], axis =1 )

dummies1= pd.get_dummies(test[mylist1], prefix= mylist1)
test.drop(mylist1, axis=1, inplace = True)
X1 = pd.concat([test,dummies1], axis =1 )


#Correlation plot 
import matplotlib.pyplot as plt

%matplotlib inline

f, ax = plt.subplots(figsize=(20, 20))
corr = X.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
            
            

X.corr()[4:5].T

#Univariate analysis on Item Visibility
sns.distplot(X['Item_Visibility'], fit=norm);


#Normalizing the skewness
X['Item_Visibility'] = np.sqrt(X['Item_Visibility'])
X1['Item_Visibility'] = np.sqrt(X1['Item_Visibility'])


#Univariate analysis on Item Visibility after standardizing
sns.distplot(X['Item_Visibility'], fit=norm);

#Univariate analysis on Item Weight
sns.distplot(X['Item_Weight'], fit=norm);

#Univariate analysis on Item MRP
sns.distplot(X['Item_MRP'], fit=norm);

#Univariate analysis on Outlet_Establishment_Year
sns.distplot(X['Outlet_Establishment_Year'], fit=norm);

del(X['Item_Identifier'])
del(X1['Item_Identifier'])


train_final=X[:]
del(train_final['Item_Outlet_Sales'])

from pandas import Series, DataFrame

# importing linear regression
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

# for cross validation
from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(train_final,X.Item_Outlet_Sales, test_size =0.3)

# training a linear regression model on train
lreg.fit(x_train,y_train)

# predicting on cv
pred_cv = lreg.predict(x_cv)

# calculating mse
mse = np.mean((pred_cv - y_cv)**2)

mse
np.sqrt(mse)

lreg.score(x_cv,y_cv)

%matplotlib inline
#Plotting error variance to check Hetroskedasticity

x_plot = plt.scatter(pred_cv, (pred_cv - y_cv), c='b')

plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')

predictors = x_train.columns

coef = Series(lreg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')







from sklearn.linear_model import ElasticNet


ENreg = ElasticNet(alpha=0.01, l1_ratio=0.95, normalize=False)

ENreg.fit(x_train,y_train)

pred_cv1 = ENreg.predict(x_cv)



mse1 = np.mean((pred_cv1 - y_cv)**2)

np.sqrt(mse1)

ENreg.score(x_cv,y_cv)


import matplotlib.pyplot as plt

%matplotlib inline

x_plot = plt.scatter(pred_cv1, (pred_cv1 - y_cv), c='b')

plt.hlines(y=0, xmin= -1000, xmax=5000)

plt.title('Residual plot')


predictors = x_train.columns

coef = Series(ENreg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')


pred_cv_final = ENreg.predict(X1)

pd.DataFrame(pred_cv_final)

X1['Predicted_sales']=pred_cv_final

X1.to_csv("/SampleSubmission.csv", sep=',', encoding='utf-8')


