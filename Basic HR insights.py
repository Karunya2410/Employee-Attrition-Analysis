import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load HR data
data = pd.read_csv("C:/Users/karunya/Documents/Guvi projects/Employee Attrition/Employee-Attrition - Employee-Attrition.csv")

# Set the title and subtitle
st.title("HR-Employee-Attrition Dashboard")


# Sidebar for filtering options
st.sidebar.header("Filter Options")
department = st.sidebar.selectbox("Select Department", data["Department"].unique())
jobrole = st.sidebar.selectbox("Select Job Role", data["JobRole"].unique())
education = st.sidebar.selectbox("Select Education Level", data["Education"].unique())
gender = st.sidebar.selectbox("Select Gender", data["Gender"].unique())

# Filter the data
filtered_data = data[(data["Department"] == department) & (data["Education"] == education)
                     & (data["Gender"] == gender) & (data["JobRole"] == jobrole)]

# Display basic statistics
st.subheader("Basic HR Statistics")
st.write("Total Employees:", len(filtered_data))
st.write("Average Age:", filtered_data["Age"].mean())

# Create two columns for charts and data
col1, col2, col3 = st.columns(3)

# Create a bar chart for employee counts by business travel in the first column
with col1:
    st.subheader("Employee Count by Business Travel")
    business_travel_counts = filtered_data["BusinessTravel"].value_counts()
    st.bar_chart(business_travel_counts)

# Create a histogram for age distribution using seaborn and Matplotlib in the second column
with col2:
    st.subheader("Age Distribution")
    plt.figure(figsize=(8, 4))
    sns.histplot(filtered_data["Age"], bins=20, kde=True)
    st.pyplot(plt)

# Calculate the attrition rate
with col3:
    attrition_count = filtered_data["Attrition"].value_counts()
    attrition_rate = attrition_count.get("Yes", 0) / len(filtered_data) * 100
    # Create a pie chart for attrition rate using Matplotlib
    st.subheader("Attrition Rate")
    labels = ['No', 'Yes']
    sizes = [len(filtered_data) - attrition_count.get("Yes", 0), attrition_count.get("Yes", 0)]
    plt.figure(figsize=(2, 2))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    st.pyplot(plt)

# Show a data table below the columns
st.subheader("Employee Data")
st.write(filtered_data)