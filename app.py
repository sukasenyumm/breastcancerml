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
    st.title("🔎 Breast Cancer Classification App (KNN and GWO)")
    st.write("""
    This app uses the **Breast Cancer Wisconsin Diagnostic dataset** to classify whether a tumor is  
    **Malignant (M)** or **Benign (B)** using a **K-Nearest Neighbors (KNN) and Grey-Wolf Optimization** model. 
    **THIS WEB IS ONLY USED FOR A DOCTOR** 

    ### Menu Guide
    - **Dashboard** → Overview of the app  
    - **Dataset** → Explore the dataset  
    - **Predict** → Input features and get predictions  
             
    ### Dataset source
    This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    """)

# Dataset
elif menu == "Dataset":
    st.title("📊 Breast Cancer Dataset")
    st.write("Here is a preview of the dataset:")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("""
    ### Attribute Information:
    1) ID number
    2) Diagnosis (M = malignant, B = benign)
    3-32)

    Ten real-valued features are computed for each cell nucleus:

    a) radius (mean of distances from center to points on the perimeter)
    b) texture (standard deviation of gray-scale values)
    c) perimeter
    d) area
    e) smoothness (local variation in radius lengths)
    f) compactness (perimeter^2 / area - 1.0)
    g) concavity (severity of concave portions of the contour)
    h) concave points (number of concave portions of the contour)
    i) symmetry
    j) fractal dimension ("coastline approximation" - 1)

    The mean, standard error and "worst" or largest (mean of the three
    largest values) of these features were computed for each image,
    resulting in 30 features. For instance, field 3 is Mean Radius, field
    13 is Radius SE, field 23 is Worst Radius.

    All feature values are recoded with four significant digits.

    Missing attribute values: none

    Class distribution: 357 benign, 212 malignant. """)

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
