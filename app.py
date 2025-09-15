# app.py
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris
import pandas as pd

# Load model and dataset
iris = load_iris()
try:
    model = joblib.load("knn_model.pkl")
except:
    model = None

# Sidebar dropdown menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Dataset", "Predict"]
)

# ---------------- Dashboard ----------------
if menu == "Dashboard":
    st.title("🌸 Iris Classification App with KNN")
    st.subheader("Welcome to the dashboard")
    st.write("""
    This app demonstrates a simple **classification task** using the Iris dataset.
    
    - **Model:** K-Nearest Neighbors (KNN)  
    - **Dataset:** Classic Iris flower dataset (150 samples, 3 species)  
    - **Goal:** Predict the species of iris flower from sepal and petal measurements.
    
    ### How to use the app:
    1. Go to the **Dataset** menu to explore the data.  
    2. Use the **Predict** menu to input flower measurements and get predictions.  
    """)

# ---------------- Dataset ----------------
elif menu == "Dataset":
    st.title("📊 Iris Dataset Overview")

    st.write("The Iris dataset is a classic dataset in machine learning, containing:")
    st.write("- 150 samples")
    st.write("- 4 features (sepal length, sepal width, petal length, petal width)")
    st.write("- 3 species: setosa, versicolor, virginica")

    # Convert dataset to DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = [iris.target_names[i] for i in iris.target]

    st.subheader("Preview of the dataset")
    st.dataframe(df.head())

    st.subheader("Class distribution")
    st.bar_chart(df["species"].value_counts())

# ---------------- Predict ----------------
elif menu == "Predict":
    st.title("🔮 Predict Iris Species")

    if model is None:
        st.error("Model not found! Please train and save the model first (run knn_iris.py).")
    else:
        # Input fields
        sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
        sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
        petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
        petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)

        if st.button("Predict"):
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)[0]
            st.success(f"Predicted species: **{iris.target_names[prediction]}**")
