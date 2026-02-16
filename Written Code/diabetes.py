import pandas as pd
import os

data_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
df = pd.read_csv(data_path)
print(df.head())
print(df.shape)