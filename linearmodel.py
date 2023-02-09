import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import mysql.connector as connection
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

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
cols = df.corr().nlargest(15, 'SalePrice')['SalePrice'].index
hm = np.corrcoef(df[cols].values.T)
# sns.heatmap(hm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

#Further traim columns correlated to each other and 
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Fireplaces']
# sns.pairplot(df[cols], height = 2.5)
print(df[cols].info())

df = df[df.SalePrice < 700000]
df = df[df.GrLivArea < 4000]
df = df[df.TotalBsmtSF < 5000]
df = df[df.TotRmsAbvGrd < 18]

fin_cols = cols.copy()
fin_cols.remove('SalePrice')
X = df[fin_cols]

print(X.info())

print(X.isnull().sum())

y = df['SalePrice']

s = MinMaxScaler()
X = pd.DataFrame(s.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)

model = XGBRegressor(learning_rate=0.05, n_jobs=3)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)

print ("Training score:",model.score(X_train,y_train), "Val Score:",model.score(X_test,y_test))

pickle.dump(model, open('xgbrmodel.pkl', 'wb'))