import numpy as np
import pandas as pd
import seaborn as sns
import mysql.connector as connection
import matplotlib.pyplot as plt

#Setup db connection and retrieve dataset
try:
    mydb = connection.connect(
        host='localhost', 
        database='ml_db', 
        user='root', 
        password='70palomino', 
        use_pure=True)
    query="SELECT * FROM house_price;"
    df=pd.read_sql(query,mydb)
    mydb.close()
except Exception as e:
    mydb.close()
    print(str(e))

#Dataset preview
print(df.head())
print(df.info())
print(df.isnull().sum())

#Use heatmap to determine which columns to use for training, pick top 15 columns based on correlation with SalePrice
cols = df.corr().nlargest(15, 'SalePrice')['SalePrice'].index
hm = np.corrcoef(df[cols].values.T)
sns.heatmap(hm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

#Further traim columns correlated to each other and 
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'Fireplaces']
sns.pairplot(df[cols], height = 2.5)
plt.show()
print(df[cols].info())