import pandas as pd

data=pd.read_csv("data.csv")
data.to_parquet("data_lr.parquet")