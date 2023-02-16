import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import mysql.connector as connection
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

#Setup db connection and retrieve dataset
try:
    mydb = connection.connect(
        host='localhost', 
        database='ml_db', 
        user='root', 
        password='70palomino', 
        use_pure=True)
    query="SELECT * FROM house_price;"
    query_test = "SELECT * FROM house_price_test"
    df=pd.read_sql(query,mydb)
    mydb.close()
except Exception as e:
    mydb.close()
    print(str(e))

#Dataset preview
print(df.head())
print(df.info())

#Use heatmap to determine which columns to use for training, pick top 15 columns based on correlation with SalePrice
cols = df.corr().nlargest(20, 'SalePrice')['SalePrice'].index
hm = np.corrcoef(df[cols].values.T)
# sns.heatmap(hm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

#Further traim columns correlated to each other and 
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Fireplaces']
# sns.pairplot(df[cols], height = 2.5)
print(df[cols].info())

#Remove significant outliers
df = df[df.SalePrice < 700000]
df = df[df.GrLivArea < 4000]
df = df[df.TotalBsmtSF < 5000]
df = df[df.TotRmsAbvGrd < 18]

df = df[cols]

# X.boxplot()
# plt.show()
print(df.shape)

#Further remove outliers
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1

    lb = Q1-1.5*IQR
    ub = Q3+1.5*IQR

    df = df[(df[col] < ub) & (df[col] > lb)]

print(df.shape)

#Finalize dataset, Feature and target
fin_cols = cols.copy()
fin_cols.remove('SalePrice')
X = df[fin_cols]

print(X.isnull().sum())

y = df['SalePrice']

s = MinMaxScaler()
df = pd.DataFrame(s.fit_transform(df), columns=df.columns)

#Split test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

#Linear Regression Training
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

MAE_lr = mean_absolute_error(y_test,y_pred_lr)
R2_lr = r2_score(y_test,y_pred_lr)
MSE_lr = mean_squared_error(y_test,y_pred_lr)

print ("Linear Regrassion:       ","R2 score - ", R2_lr, "MAE - ", MAE_lr, "MSE - ", MSE_lr)

#Random Forest Regression Training
model_rfr = RandomForestRegressor()
model_rfr.fit(X_train, y_train)

y_pred_rfr = model_rfr.predict(X_test)

MAE_rfr = mean_absolute_error(y_test,y_pred_rfr)
R2_rfr = r2_score(y_test,y_pred_rfr)
MSE_rfr = mean_squared_error(y_test,y_pred_rfr)

print ("Random Forest Regressor: ","R2 score - ", R2_rfr, "MAE - ", MAE_rfr, "MSE - ", MSE_rfr)

#XGBoost Regressor training
model_xgb = XGBRegressor(learning_rate=0.05, n_jobs=3)
model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_xgb = model_xgb.predict(X_test)


MAE_xgb = mean_absolute_error(y_test,y_pred_xgb)
R2_xgb = r2_score(y_test,y_pred_xgb)
MSE_xgb = mean_squared_error(y_test,y_pred_xgb)

print ("XGBoost Regressor:       ","R2 score - ", R2_xgb, "MAE - ", MAE_xgb, "MSE - ", MSE_xgb)
#Comparing the Metrics XGBoot Regressor have the highest R2 score which is more accurate in predication than the other two models and the Lowest Mean Squared error which means this model can better represent the dataset.

pickle.dump(model_xgb, open('xgbrmodel.pkl', 'wb'))