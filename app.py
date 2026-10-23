
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Cleaned_Chocolate_Sales.csv")


# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Check required columns
required = ["country", "product", "date", "amount", "boxes_shipped"]

if not set(required).issubset(df.columns):
    raise ValueError(
        f"CSV must contain these columns: {required}"
    )

# Clean data
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["boxes_shipped"] = pd.to_numeric(
    df["boxes_shipped"], errors="coerce"
)

df = df.dropna(subset=required).copy()
df = df[
    (df["amount"] >= 0) &
    (df["boxes_shipped"] >= 0)
].copy()

# Extract date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

features = [
    "country", "product", "boxes_shipped",
    "year", "month", "day"
]

X = df[features]
y = df["amount"]

# Preprocess text categories and train regression model
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            ["country", "product"]
        ),
        (
            "numbers",
            "passthrough",
            ["boxes_shipped", "year", "month", "day"]
        )
    ]
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

model.fit(X, y)

# Dashboard statistics
total_sales = round(df["amount"].sum(), 2)
total_boxes = int(df["boxes_shipped"].sum())
best_product = (
    df.groupby("product")["amount"]
    .sum()
    .idxmax()
)

country_sales = (
    df.groupby("country")["amount"]
    .sum()
    .sort_values(ascending=False)
)

product_sales = (
    df.groupby("product")["amount"]
    .sum()
    .sort_values(ascending=False)
)

monthly_sales = (
    df.assign(month_label=df["date"].dt.to_period("M").astype(str))
    .groupby("month_label")["amount"]
    .sum()
    .sort_index()
)

@app.route("/")
def home():
    return render_template(
        "index.html",
        total_sales=total_sales,
        total_boxes=total_boxes,
        best_product=best_product
    )




@app.route("/analysis")
def analysis():
    # 1. Sales by country
    country_sales = (
        df.groupby("country")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    # 2. Sales by product
    product_sales = (
        df.groupby("product")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    # 3. Monthly sales trend
    monthly_sales = (
        df.assign(
            month_label=df["date"].dt.to_period("M").astype(str)
        )
        .groupby("month_label")["amount"]
        .sum()
        .sort_index()
    )

    return render_template(
        "analysis.html",
        countries=country_sales.index.tolist(),
        country_values=country_sales.values.tolist(),
        products=product_sales.index.tolist(),
        product_values=product_sales.values.tolist(),
        months=monthly_sales.index.tolist(),
        monthly_values=monthly_sales.values.tolist()
    )



@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None

    countries = sorted(df["country"].unique().tolist())
    products = sorted(df["product"].unique().tolist())

    if request.method == "POST":
        try:
            country = request.form["country"]
            product = request.form["product"]
            boxes = float(request.form["boxes_shipped"])
            date = pd.to_datetime(request.form["date"], errors="raise")

            if country not in countries or product not in products:
                raise ValueError("Select a valid country and product.")

            if not np.isfinite(boxes) or boxes < 0:
                raise ValueError("Boxes shipped must be a non-negative number.")

            input_data = pd.DataFrame([{
                "country": country,
                "product": product,
                "boxes_shipped": boxes,
                "year": date.year,
                "month": date.month,
                "day": date.day
            }])

            result = round(
                max(0, float(model.predict(input_data)[0])), 2
            )

        except (ValueError, TypeError):
            error = "Please check your date, product, country and box quantity."

    return render_template(
        "prediction.html",
        countries=countries,
        products=products,
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
