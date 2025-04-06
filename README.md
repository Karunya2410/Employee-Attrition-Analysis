# 🧠 Employee Attrition Prediction and Analysis

This Streamlit application enables HR professionals and analysts to explore employee data, identify key factors contributing to attrition, and predict which employees are at risk of leaving. It uses machine learning and data visualization techniques to deliver actionable insights for workforce retention strategies.

---

## 📊 Features

### ✅ Data Collection & Preprocessing
- Utilizes a structured employee dataset (demographics, job info, performance).
- Handles missing values, categorical encoding, and outlier detection.
  
### 🔍 Exploratory Data Analysis (EDA)
- Visualizes attrition patterns across departments, roles, job satisfaction, salary levels, etc.
- Correlation heatmaps and boxplots to detect influential variables.

### 🧠 Feature Engineering
- Creates new features such as:
  - Tenure categories
  - Performance bands
  - Engagement scores

### 🤖 Machine Learning Model
- Predictive modeling using:
  - Logistic Regression
  - Decision Trees
  - Random Forests
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion matrix

### 📈 Interactive Dashboard
- Built with [Streamlit](https://streamlit.io/)
- Allows HR teams to:
  - Visualize attrition trends
  - Explore model predictions
  - Filter data by job role, department, etc.
  - Download predictions

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/employee-attrition-app.git
cd employee-attrition-app

2. Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
