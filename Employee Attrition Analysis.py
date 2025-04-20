import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import xgboost as xgb


file_path = "C:/Users/karunya/Documents/Guvi projects/Employee Attrition/Cleaned_Employee_Attrition.csv"
df = pd.read_csv(file_path)

st.title("Employee Insights Prediction Dashboard")

st.sidebar.title("Configure Model")
target_option = st.sidebar.selectbox("Select Prediction Target", ["Attrition", "JobSatisfaction"])

st.subheader("Exploratory Data Analysis (EDA)")

if target_option == "Attrition":
    # **1. Attrition vs Monthly Income (Boxplot)**
    st.write("### 1. How Attrition Relates to Monthly Income")
    st.write("""
    This boxplot illustrates how employees with different monthly incomes are distributed between those who stayed at the company and those who left (Attrition).
    It can give us insights into whether employees with higher or lower incomes are more likely to leave the company.
    """)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df['Attrition'], y=df['MonthlyIncome'], palette="coolwarm", ax=ax)
    ax.set_title("Monthly Income Distribution by Attrition")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Monthly Income")
    st.pyplot(fig)

    
    st.write("### 2. How Job Satisfaction Affects Attrition")
    st.write("""
    This bar plot shows the distribution of job satisfaction levels for employees who stayed vs those who left. Job satisfaction is measured on a scale of 1 (low) to 4 (high).
    From this plot, you can see how the level of job satisfaction affects employees' decision to stay or leave the company.
    """)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x=df['JobSatisfaction'], hue=df['Attrition'], palette="coolwarm", ax=ax)
    ax.set_title("Attrition by Job Satisfaction Level")
    ax.set_xlabel("Job Satisfaction (1 = Low, 4 = High)")
    ax.set_ylabel("Count")
    ax.legend(title="Attrition", labels=["No", "Yes"])
    st.pyplot(fig)

elif target_option == "JobSatisfaction":
    # **Job Satisfaction vs Monthly Income (Boxplot)**
    st.write("### How Job Satisfaction Relates to Monthly Income")
    st.write("""
    This boxplot shows how job satisfaction levels are distributed across different income groups.
    Employees with higher incomes are likely to have higher job satisfaction, but this may vary.
    """)
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x=df['JobSatisfaction'], y=df['MonthlyIncome'], palette="coolwarm", ax=ax)
    ax.set_title("Monthly Income by Job Satisfaction Level")
    ax.set_xlabel("Job Satisfaction")
    ax.set_ylabel("Monthly Income")
    st.pyplot(fig)

# Preprocessing
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature & Target selection
if target_option == "Attrition":
    features = ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction',
                'YearsAtCompany', 'MaritalStatus', 'OverTime']
    target = 'Attrition'
    is_classification = True

elif target_option == "JobSatisfaction":
    features = ['Age', 'Department', 'MonthlyIncome', 'YearsAtCompany',
                'OverTime', 'JobRole']
    target = 'JobSatisfaction'
    is_classification = False

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Selection based on target option
if target_option == "Attrition":
    model = DecisionTreeClassifier(random_state=42)
elif target_option == "JobSatisfaction":
    model = LinearRegression()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# User input section (Advanced)
st.write(f"### Predict {target_option} for an Individual")
st.markdown("Provide inputs below to make a prediction.")

user_input = []
with st.form(key="prediction_form"):
    for feat in features:
        if feat in label_encoders:
            options = df[feat].unique()
            decoded_options = label_encoders[feat].inverse_transform(sorted(options))
            val = st.selectbox(f"Select {feat}", decoded_options)
            encoded_val = label_encoders[feat].transform([val])[0]
        elif feat == 'Age':
            encoded_val = st.slider("Select Age", int(df[feat].min()), int(df[feat].max()), int(df[feat].median()))
        elif feat == 'MonthlyIncome':
            encoded_val = st.number_input("Enter Monthly Income", min_value=0, max_value=100000, value=int(df[feat].median()), step=500)
        elif 'Years' in feat:
            encoded_val = st.number_input(f"Enter {feat}", min_value=0, max_value=40, value=int(df[feat].median()), step=1)
        else:
            encoded_val = st.number_input(f"Enter {feat}", value=float(df[feat].median()))

        user_input.append(encoded_val)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    if target_option == "JobSatisfaction":
        st.success(f"Predicted Job Satisfaction: {prediction:.2f}")
    else:
        st.success(f"Predicted Attrition: {'Yes' if prediction else 'No'}")
