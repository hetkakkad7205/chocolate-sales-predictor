
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Load cleaned dataset
df = pd.read_csv("Cleaned_Chocolate_Sales.csv")

# 2. Extract date features
df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# 3. Select input and target columns
X = df[
    ["Country", "Product", "Boxes Shipped",
     "Year", "Month", "Day"]
]

y = df["Amount"]

# 4. Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Country", "Product"],
    drop_first=True,
    dtype=int
)

# Save feature names for Flask prediction
model_columns = X.columns.tolist()

# 5. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Create Linear Regression model
model = LinearRegression()

# 7. Train model
model.fit(X_train, y_train)

# 8. Test model
y_pred = model.predict(X_test)

# 9. Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.4f}")


# 10. Actual vs Predicted graph
if __name__ == "__main__":

    plt.figure(figsize=(8, 5))

    plt.scatter(y_test, y_pred, alpha=0.6)

    plt.xlabel("Actual Amount")
    plt.ylabel("Predicted Amount")
    plt.title("Actual vs Predicted Chocolate Sales")

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--"
    )

    plt.tight_layout()
    plt.show()
