import pandas as pd
import numpy as np

df=pd.read_csv("Chocolate Sales.csv")
print(df.head()) # this will print first top 5 records
print(df.info())# this will print info about datatype
print(df.shape)
