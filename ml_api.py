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
    Firplaces: int

@app.post('/')
async def price_endpoint(item:HouseData):
    return item