import numpy as np
import pandas as pd
#import matplotlib.plyplot as plt
import seaborn as sns

data = pd.read_csv("Cleaned_Chocolate_Sales.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.columns)

print("Missing values:")
print(data.isnull().sum())

print("Duplicate rows:", data.duplicated().sum())

print("Total Sales:", data["Amount"].sum())
print("Average Sales:", data["Amount"].mean())
print("Highest Sale:", data["Amount"].max())
print("Lowest Sale:", data["Amount"].min())

print("Total Boxes Shipped:", data["Boxes Shipped"].sum())
print("Average Boxes Shipped:", data["Boxes Shipped"].mean())

country_sales = data.groupby("Country")["Amount"].sum().sort_values()
print(country_sales)

