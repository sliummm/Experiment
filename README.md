# Experiment
Machine Learning API for house price prediction

- POST: takes a JSON object:
{
    "OverallQual": int,
    "GrLivArea": int,
    "GarageCars": int,
    "TotalBsmtSF": int,
    "FullBath": int,
    "TotRmsAbvGrd": int,
    "YearBuilt": int,
    "YearRemodAdd": int,
    "Fireplaces": int
}

return {"SalePrice":int}
