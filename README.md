# 🍫 Chocolate Sales Predictor

A Data Science and Machine Learning project that analyzes chocolate sales data and predicts sales using **Linear Regression**. The project combines data cleaning, exploratory data analysis (EDA), data visualization, and predictive modeling to understand sales patterns and support data-driven decisions.

## 📌 Project Overview

The Chocolate Sales Predictor uses historical chocolate sales data to analyze sales performance and estimate sales based on selected input features.

## ✨ Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Sales trends and graphical visualizations
- Feature engineering using date-related information, if applicable
- Sales prediction using Linear Regression
- Model evaluation using MAE, MSE, and R² score
- Web-based interface for interacting with the prediction system

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit or Flask, depending on the deployed interface
- HTML, CSS and JavaScript, if using the Flask frontend
- Jupyter Notebook

## 📂 Project Structure

```text
Chocolate-sales-predictor/
├── app.py
├── model.py
├── data_cleaning.py
├── data.csv
├── cleaned_data.csv
├── eda_analysis.ipynb
├── requirements.txt
├── README.md
├── templates/
└── static/
    ├── css/
    └── js/
```

## ⚙️ Machine Learning Model

**Algorithm:** Linear Regression

The model learns relationships between the selected input features and the sales target to estimate sales values.

### Evaluation Metrics

- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **R² Score:** Measures how well the model explains variation in the target variable

Actual performance values should be added after running the model.

## 🚀 How to Run Locally

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Open the project folder:

   ```bash
   cd Chocolate-sales-predictor
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   For Streamlit:

   ```bash
   streamlit run app.py
   ```

   For Flask, use the startup command defined by your application.

## 📊 Project Workflow

1. Load the original sales dataset.
2. Clean and preprocess the data.
3. Perform EDA and visualize sales patterns.
4. Prepare features and the target variable.
5. Train and evaluate the Linear Regression model.
6. Display predictions and relevant visualizations through the web interface.

## 🌐 Live Demo

Add your deployed application URL here after deployment.

**Live Demo:** YOUR_DEPLOYED_APP_URL

## 👨‍💻 Author

Het Kakkad(Internship Project)

## 📄 License

This project is developed for educational and internship purposes.
