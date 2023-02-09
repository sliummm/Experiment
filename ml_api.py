import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    TotRmsAbvGrd: int
    YearBuilt: int
    YearRemodAdd: int
    Fireplaces: int

with open('xgbrmodel.pkl', 'rb') as xgbr:
    model = pickle.load(xgbr)

@app.post('/')
async def price_endpoint(item:HouseData):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())

    y_pred = model.predict(df)
    return {"SalePrice": int(y_pred)}