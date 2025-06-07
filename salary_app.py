import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping short forms to full forms
EXP_LEVEL_MAP = {'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}
EMP_TYPE_MAP = {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}
COMP_SIZE_MAP = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
COUNTRY_MAP = {
    'US': 'United States', 'IN': 'India', 'GB': 'United Kingdom', 'CA': 'Canada',
    'DE': 'Germany', 'FR': 'France', 'ES': 'Spain', 'NL': 'Netherlands',
    'AU': 'Australia', 'BR': 'Brazil'
}

REV_EXP_LEVEL_MAP = {v: k for k, v in EXP_LEVEL_MAP.items()}
REV_EMP_TYPE_MAP = {v: k for k, v in EMP_TYPE_MAP.items()}
REV_COMP_SIZE_MAP = {v: k for k, v in COMP_SIZE_MAP.items()}
REV_COUNTRY_MAP = {v: k for k, v in COUNTRY_MAP.items()}

# Load dataset
df = pd.read_csv("final_salaries.csv")

# Preprocess data
def preprocess_data(df):
    df_processed = df.drop(columns=["salary", "salary_currency", "salary_in_usd"])
    categorical_cols = df_processed.select_dtypes(include='object').columns
    encoder = OrdinalEncoder()
    df_processed[categorical_cols] = encoder.fit_transform(df_processed[categorical_cols])
    return df_processed, df['salary_in_usd'], encoder, categorical_cols

# Apply readable mappings for display
def apply_mappings(df):
    df = df.copy()
    df['experience_level'] = df['experience_level'].map(EXP_LEVEL_MAP)
    df['employment_type'] = df['employment_type'].map(EMP_TYPE_MAP)
    df['company_size'] = df['company_size'].map(COMP_SIZE_MAP)
    df['employee_residence'] = df['employee_residence'].map(COUNTRY_MAP).fillna(df['employee_residence'])
    df['company_location'] = df['company_location'].map(COUNTRY_MAP).fillna(df['company_location'])
    return df

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Streamlit App
st.set_page_config(layout="wide")
st.title("Salary Prediction App")

# Sidebar navigation
session = st.sidebar.radio("Choose a session", ["Dataset", "Graphs", "Prediction"])

# Load and preprocess
data = df.copy()
X, y, encoder, cat_cols = preprocess_data(data)
model = train_model(X, y)
feature_order = X.columns.tolist()  # Save column order used in training

# Session 1: Dataset
if session == "Dataset":
    st.header("📊 Dataset Overview")
    st.write(apply_mappings(data).head())
    st.write("Shape:", data.shape)

# Session 2: Graphs
elif session == "Graphs":
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Experience Level Distribution")
        sns.countplot(x="experience_level", data=apply_mappings(data))
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.subheader("Company Size Distribution")
        sns.countplot(x="company_size", data=apply_mappings(data))
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("Salary in USD vs. Remote Ratio")
    sns.scatterplot(x="remote_ratio", y="salary_in_usd", data=apply_mappings(data))
    st.pyplot(plt.gcf())
    plt.clf()

# Session 3: Prediction
elif session == "Prediction":
    st.header("💼 Salary Prediction")

    input_data = {}
    input_data['work_year'] = st.number_input("Work Year", min_value=2020, max_value=2025, value=2024)
    exp_level = st.selectbox("Experience Level", list(EXP_LEVEL_MAP.values()))
    emp_type = st.selectbox("Employment Type", list(EMP_TYPE_MAP.values()))
    job_title = st.selectbox("Job Title", data['job_title'].unique())
    residence = st.selectbox("Employee Residence", list(COUNTRY_MAP.values()))
    input_data['remote_ratio'] = st.slider("Remote Ratio", 0, 100, 0)
    company_location = st.selectbox("Company Location", list(COUNTRY_MAP.values()))
    comp_size = st.selectbox("Company Size", list(COMP_SIZE_MAP.values()))

    # Reverse mapping to short forms
    input_data['experience_level'] = REV_EXP_LEVEL_MAP[exp_level]
    input_data['employment_type'] = REV_EMP_TYPE_MAP[emp_type]
    input_data['job_title'] = job_title
    input_data['employee_residence'] = REV_COUNTRY_MAP[residence]
    input_data['company_location'] = REV_COUNTRY_MAP[company_location]
    input_data['company_size'] = REV_COMP_SIZE_MAP[comp_size]

    # Convert to DataFrame and encode
    input_df = pd.DataFrame([input_data])
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])
    input_df = input_df.reindex(columns=feature_order)  # Match training order

    prediction = model.predict(input_df)[0]
    st.subheader(f"\n\n💰 Predicted Salary in USD: ${int(prediction):,}")