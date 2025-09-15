import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
scaler, model = joblib.load("best_knn_gwo.pkl")

# Load dataset for Dataset page
url = "https://raw.githubusercontent.com/minikku/Public-Datasets/refs/heads/main/breast_cancer_wisconsin_diagnostic.csv"
df = pd.read_csv(url)
df = df.drop(columns=["id"])
df = df.drop(columns=["Unnamed: 32"], errors="ignore")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Choose a Menu",
    ["Dashboard", "Dataset", "Predict"]
)

# Dashboard
if menu == "Dashboard":
    st.title("🔎 Breast Cancer Classification App (KNN)")
    st.write("""
    This app uses the **Breast Cancer Wisconsin Diagnostic dataset** to classify whether a tumor is  
    **Malignant (M)** or **Benign (B)** using a **K-Nearest Neighbors (KNN) and Grey-Wolf Optimization** model.  

    ### Menu Guide
    - **Dashboard** → Overview of the app  
    - **Dataset** → Explore the dataset  
    - **Predict** → Input features and get predictions  
    """)

# Dataset
elif menu == "Dataset":
    st.title("📊 Breast Cancer Dataset")
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)

# Prediction
elif menu == "Predict":
    st.title("🩺 Predict Breast Cancer")

    st.write("Please input the following features:")

    # Collect inputs dynamically
    input_data = {}
    for col in df.drop(columns=["diagnosis"]).columns:
        input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    # Convert to dataframe
    input_df = pd.DataFrame([input_data])

    # Scale input
    input_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"Prediction: **{'Malignant' if prediction == 'M' else 'Benign'}**")
