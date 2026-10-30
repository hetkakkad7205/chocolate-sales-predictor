
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Import trained model from your training file
from linear_regression_model import model, model_columns

app = Flask(__name__)


# ==========================================
# 1. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv("Cleaned_Chocolate_Sales.csv")

# Standardize column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Convert numerical columns
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

df["boxes_shipped"] = pd.to_numeric(
    df["boxes_shipped"], errors="coerce"
)

# Remove invalid records
df = df.dropna(
    subset=["country", "product", "date",
            "amount", "boxes_shipped"]
).copy()

df = df[
    (df["amount"] >= 0) &
    (df["boxes_shipped"] >= 0)
].copy()


# ==========================================
# 2. DASHBOARD STATISTICS
# ==========================================

total_sales = round(df["amount"].sum(), 2)

total_boxes = int(df["boxes_shipped"].sum())

best_product = (
    df.groupby("product")["amount"]
    .sum()
    .idxmax()
)


# ==========================================
# 3. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        total_sales=total_sales,
        total_boxes=total_boxes,
        best_product=best_product
    )


# ==========================================
# 4. SALES ANALYSIS PAGE
# ==========================================

@app.route("/analysis")
def analysis():

    # Sales by country
    country_sales = (
        df.groupby("country")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    # Sales by product
    product_sales = (
        df.groupby("product")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    # Monthly sales
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


# ==========================================
# 5. CHOCOLATE SALES PREDICTION
# ==========================================

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    result = None
    error = None

    countries = sorted(df["country"].unique().tolist())
    products = sorted(df["product"].unique().tolist())

    if request.method == "POST":

        try:

            # Get user input
            country = request.form["country"]
            product = request.form["product"]

            boxes = float(request.form["boxes_shipped"])

            date = pd.to_datetime(
                request.form["date"],
                errors="raise"
            )

            # Validate country and product
            if country not in countries:
                raise ValueError("Invalid country")

            if product not in products:
                raise ValueError("Invalid product")

            # Validate boxes
            if not np.isfinite(boxes) or boxes < 0:
                raise ValueError("Invalid box quantity")

            # Create input DataFrame
            input_data = pd.DataFrame([{
                "Country": country,
                "Product": product,
                "Boxes Shipped": boxes,
                "Year": date.year,
                "Month": date.month,
                "Day": date.day
            }])

            # Convert categorical values into dummy columns
            input_data = pd.get_dummies(
                input_data,
                columns=["Country", "Product"],
                drop_first=True,
                dtype=int
            )

            # Match training columns
            input_data = input_data.reindex(
                columns=model_columns,
                fill_value=0
            )

            # Predict using imported trained model
            prediction_value = model.predict(input_data)[0]

            result = round(max(0, float(prediction_value)), 2)

        except (ValueError, TypeError, KeyError):
            error = (
                "Please check your date, product, "
                "country and box quantity."
            )

    return render_template(
        "prediction.html",

        countries=countries,
        products=products,

        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
