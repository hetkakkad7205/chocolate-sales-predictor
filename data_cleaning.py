
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("Chocolate Sales.csv")

# Step 1: Drop irrelevant columns
data = df.drop(columns=["Sales Person"])

# Step 2: Handle missing data
print("\nMissing values before cleaning:")
print(data.isnull().sum())

data = data.dropna(
    subset=["Country", "Product", "Date", "Amount", "Boxes Shipped"]
)

# Step 3: Fix inconsistent values
data["Country"] = data["Country"].replace({
    "India": "INDIA"
})

# Step 4: Standardize text
data["Country"] = data["Country"].str.strip().str.upper()
data["Product"] = data["Product"].str.strip().str.title()

# Step 5: Convert data types
data["Date"] = pd.to_datetime(
    data["Date"],
    format="%d-%b-%y"
)

data["Amount"] = pd.to_numeric(
    data["Amount"].astype(str).str.replace(r"[$,]", "", regex=True))


data["Boxes Shipped"] = pd.to_numeric(
    data["Boxes Shipped"])

# Remove rows with invalid converted values
data = data.dropna(
    subset=["Date", "Amount", "Boxes Shipped"]
)

# Step 6: Remove duplicate records
data = data.drop_duplicates()

# Step 7: Display cleaned dataset
print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nCleaned Dataset:")
print(data)

# Step 8: Save cleaned dataset
data.to_csv(
    "D:/SqRock_IT_Solutions_Internship/Phase-1/Task-1_Sales_Analysis_Predictor/Cleaned_Chocolate_Sales.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
