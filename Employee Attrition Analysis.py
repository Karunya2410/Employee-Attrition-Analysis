import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load the data
file_path = "C:/Users/karunya/Documents/Guvi projects/Employee Attrition/Employee-Attrition - Employee-Attrition.csv"
df = pd.read_csv(file_path)

# Title
st.title("Employee Analytics Prediction Dashboard")

# Data Preprocessing
st.subheader("Data Preprocessing")
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# Sidebar for model selection
st.sidebar.title("Model Configuration")
target_option = st.sidebar.selectbox("Select Prediction Target", 
                                         ["Attrition", "JobSatisfaction", "PerformanceRating"])

model_option = st.sidebar.selectbox("Select Model", 
                                        ["Random Forest", "Logistic Regression", "Decision Tree"])

# Define features and target for each prediction type
if target_option == "Attrition":
    features = ['Age', 'Department', 'MonthlyIncome', 'JobSatisfaction',
                    'YearsAtCompany', 'MaritalStatus', 'OverTime']
    target = 'Attrition'

elif target_option == "JobSatisfaction":
    features = ['Age', 'Gender', 'Department', 'MonthlyIncome', 'JobRole',
                    'TrainingTimesLastYear', 'RelationshipSatisfaction']
    target = 'JobSatisfaction'

else:
    features = ['Education', 'JobInvolvement', 'JobLevel', 'MonthlyIncome',
                    'YearsAtCompany', 'YearsInCurrentRole']
    target = 'PerformanceRating'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model selection
if model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show metrics
st.write(f"### Classification Report for Predicting {target_option}")
st.text(classification_report(y_test, y_pred))

# User input for prediction
st.write(f"### Make Individual {target_option} Prediction")
user_input = []
for feat in features:
    val = st.number_input(f"Enter value for {feat}", value=float(df[feat].median()))
    user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted {target_option}: {prediction}")

