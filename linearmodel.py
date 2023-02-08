import numpy as np
import pandas as pd
import mysql.connector as connection

try:
    mydb = connection.connect(
        host='localhost', 
        database='housepriceprediction', 
        user='root', 
        password='70palomino', 
        use_pure=True)
    query="SELECT * FROM house_price;"
    df=pd.read_sql(query,mydb)
    mydb.close()
except Exception as e:
    mydb.close()
    print(str(e))

print(df.head())
print(df.info())
print(df.isnull().sum())
