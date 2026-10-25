
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Chocolate Sales Predictor",
    page_icon="🍫",
    layout="wide"
)


# ---------------- LOAD AND CLEAN DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Chocolate_Sales.csv")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    required = [
        "country", "product", "date",
        "amount", "boxes_shipped"
    ]

    if not set(required).issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )
    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )
    df["boxes_shipped"] = pd.to_numeric(
        df["boxes_shipped"], errors="coerce"
    )

    df = df.dropna(subset=required).copy()

    df = df[
        (df["amount"] >= 0)
        & (df["boxes_shipped"] >= 0)
    ].copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    return df


try:
    df = load_data()

    if df.empty:
        st.error("No valid records found in the dataset.")
        st.stop()

except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


# ---------------- TRAIN MODEL ----------------
features = [
    "country", "product", "boxes_shipped",
    "year", "month", "day"
]

X = df[features]
y = df["amount"]

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


# ---------------- DASHBOARD STATISTICS ----------------
total_sales = df["amount"].sum()
total_boxes = df["boxes_shipped"].sum()

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
    df.assign(
        month_label=df["date"].dt.to_period("M").astype(str)
    )
    .groupby("month_label")["amount"]
    .sum()
    .sort_index()
)


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🍫 Chocolate Sales")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Sales Analysis", "Sales Prediction", "Dataset"]
)


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.title("🍫 Chocolate Sales Predictor")
    st.write(
        "Analyze chocolate sales performance "
        "and predict sales using Machine Learning."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{total_sales:,.2f}")
    col2.metric("Total Boxes Shipped", f"{total_boxes:,.0f}")
    col3.metric("Best-Selling Product", str(best_product))

    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly_sales)

    st.subheader("Top Products by Sales")
    st.bar_chart(product_sales.head(10))


# ---------------- SALES ANALYSIS ----------------
elif page == "Sales Analysis":
    st.title("📊 Sales Analysis")

    st.subheader("Sales by Country")
    st.bar_chart(country_sales)

    st.subheader("Sales by Product")
    st.bar_chart(product_sales)

    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly_sales)

    st.subheader("Sales Summary")
    st.dataframe(
        df.groupby("country")["amount"]
        .agg(["sum", "mean", "count"])
        .rename(columns={
            "sum": "Total Sales",
            "mean": "Average Sales",
            "count": "Records"
        }),
        use_container_width=True
    )


# ---------------- SALES PREDICTION ----------------
elif page == "Sales Prediction":
    st.title("🔮 Predict Chocolate Sales")

    st.write(
        "Select a country, product, date and boxes shipped "
        "to estimate the sales amount."
    )

    countries = sorted(df["country"].unique().tolist())
    products = sorted(df["product"].unique().tolist())

    with st.form("prediction_form"):
        country = st.selectbox("Select Country", countries)
        product = st.selectbox("Select Product", products)

        boxes = st.number_input(
            "Boxes Shipped",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        selected_date = st.date_input("Select Date")

        submitted = st.form_submit_button("Predict Sales")

    if submitted:
        input_data = pd.DataFrame([{
            "country": country,
            "product": product,
            "boxes_shipped": boxes,
            "year": selected_date.year,
            "month": selected_date.month,
            "day": selected_date.day
        }])

        prediction = float(model.predict(input_data)[0])
        prediction = max(0, prediction)

        st.success(f"Predicted Sales Amount: {prediction:,.2f}")

        st.caption(
            "This is a model estimate based on historical data, "
            "not a guaranteed future sales amount."
        )


# ---------------- DATASET ----------------
elif page == "Dataset":
    st.title("📁 Chocolate Sales Dataset")

    st.write(f"Total valid records: {len(df)}")

    st.dataframe(df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)
