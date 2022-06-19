import sklearn
import numpy as np
import pandas as pd
#%%
from sklearn import datasets
ds = datasets.load_boston()
Boston = pd.DataFrame(ds.data, columns=(ds.feature_names))
Boston["Price"] = ds.target
#%%
from sklearn.linear_model import *
x = Boston.iloc[:,:-1]
y = Boston.Price
model = Lasso(alpha = 0.05, normalize=(True))
model.fit(x,y)
c = model.coef_
#%%
from sklearn.model_selection import train_test_split
x_new = x.iloc[:,[5,10,11,12]]
xtrain,xtest,ytrain,ytest = train_test_split(x_new,y, test_size=0.3, random_state=42)
model2 = LinearRegression().fit(xtrain,ytrain)
ypred = model2.predict(xtest)
#%%
a = model2.intercept_
b = model2.coef_
#%%
def Price_Prediction(RM,PTRATIO,B,LSTAT):
    df =  np.array([RM,PTRATIO,B,LSTAT])
    PRICE = a + (np.reshape(df,(1,-1)).dot(b))
    return PRICE
#%%
RM = float(input("Please Enter  average number of rooms per dwelling :"))
PTRATIO = float(input("Please Enter pupil-teacher ratio by town :"))
B = float(input("Please Enter BK :"))
LSTAT = float(input("Please Enter % lower status of the population :"))
print(Price_Prediction(RM, PTRATIO, B, LSTAT))
#%%
