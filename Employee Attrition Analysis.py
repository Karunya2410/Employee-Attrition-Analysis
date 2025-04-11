import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error

# Load data
file_path = "C:/Users/karunya/Documents/Guvi projects/Employee Attrition/Cleaned_Employee_Attrition.csv"
df = pd.read_csv(file_path)

# Title
st.title("Employee Insights Prediction Dashboard")

# Preprocessing
st.subheader("Data Preprocessing")
df.dropna(inplace=True)
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Sidebar
st.sidebar.title("Configure Model")
target_option = st.sidebar.selectbox("Select Prediction Target", ["Attrition", "PerformanceRating", "PromotionLikelihood"])
model_option = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Decision Tree"])

# Feature & Target selection
if target_option == "Attrition":
    features = ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction',
                'YearsAtCompany', 'MaritalStatus', 'OverTime']
    target = 'Attrition'
    is_classification = True

elif target_option == "PerformanceRating":
    features = ['Education', 'JobInvolvement', 'JobLevel', 'MonthlyIncome',
                'YearsAtCompany', 'YearsInCurrentRole']
    target = 'PerformanceRating'
    is_classification = True

else:  # PromotionLikelihood
    features = ['JobLevel', 'TotalWorkingYears', 'YearsInCurrentRole',
                'PerformanceRating', 'Education']
    target = 'YearsSinceLastPromotion'
    is_classification = False

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model selection
if is_classification:
    if model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = DecisionTreeClassifier()
else:
    model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output metrics
if is_classification:
    st.write(f"### Classification Report for {target_option}")
    st.text(classification_report(y_test, y_pred))
else:
    st.write(f"### Mean Squared Error for {target_option}")
    st.write(mean_squared_error(y_test, y_pred))

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
    if is_classification and target in label_encoders:
        prediction = label_encoders[target].inverse_transform([int(prediction)])[0]
    st.success(f"Predicted {target_option}: {prediction}")
