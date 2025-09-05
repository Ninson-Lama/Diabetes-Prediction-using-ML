# diabetes_app.py - Final Fixed Version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ----------------- Load Model and Encoders -----------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("best_rf_random.joblib")
    # Expected features (must match training order exactly)
    features = ['Age', 'Sex', 'Ethnicity', 'BMI', 'Waist_Circumference',
                'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic',
                'Blood_Pressure_Diastolic', 'Cholesterol_Total', 'Cholesterol_HDL',
                'Cholesterol_LDL', 'GGT', 'Serum_Urate', 'Physical_Activity_Level',
                'Dietary_Intake_Calories', 'Alcohol_Consumption', 'Smoking_Status',
                'Family_History_of_Diabetes', 'Previous_Gestational_Diabetes']
    return model, features

@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)

model, feature_columns = load_model_and_features()
label_encoders = load_encoders()

# ----------------- App Title -----------------
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk using the trained Random Forest model.")

# ----------------- User Input -----------------
def user_input():
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", label_encoders.get('Sex', None).classes_ if 'Sex' in label_encoders else ["Male","Female"])
    ethnicity = st.selectbox("Ethnicity", label_encoders.get('Ethnicity', None).classes_ if 'Ethnicity' in label_encoders else ["GroupA","GroupB"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    waist = st.number_input("Waist Circumference (cm)", 40.0, 200.0, 90.0)
    glucose = st.number_input("Fasting Blood Glucose (mg/dL)", 50.0, 300.0, 100.0)
    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
    bp_sys = st.number_input("Blood Pressure Systolic", 80.0, 200.0, 120.0)
    bp_dia = st.number_input("Blood Pressure Diastolic", 50.0, 120.0, 80.0)
    chol_total = st.number_input("Cholesterol Total", 100.0, 400.0, 180.0)
    chol_hdl = st.number_input("Cholesterol HDL", 20.0, 100.0, 50.0)
    chol_ldl = st.number_input("Cholesterol LDL", 50.0, 250.0, 100.0)
    ggt = st.number_input("GGT", 5.0, 300.0, 40.0)
    urate = st.number_input("Serum Urate", 2.0, 10.0, 5.0)
    activity = st.selectbox("Physical Activity Level", ["Low","Medium","High"])
    calories = st.number_input("Dietary Intake Calories", 1000.0, 5000.0, 2500.0)
    alcohol = st.selectbox("Alcohol Consumption", label_encoders.get('Alcohol_Consumption', None).classes_ if 'Alcohol_Consumption' in label_encoders else ["Yes","No"])
    smoking = st.selectbox("Smoking Status", label_encoders.get('Smoking_Status', None).classes_ if 'Smoking_Status' in label_encoders else ["Never","Former","Current"])
    family = st.selectbox("Family History of Diabetes", ["Yes","No"])
    gestational = st.selectbox("Previous Gestational Diabetes", [0,1])

    df_input = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'Ethnicity': [ethnicity], 'BMI': [bmi],
        'Waist_Circumference': [waist], 'Fasting_Blood_Glucose': [glucose],
        'HbA1c': [hba1c], 'Blood_Pressure_Systolic': [bp_sys],
        'Blood_Pressure_Diastolic': [bp_dia], 'Cholesterol_Total': [chol_total],
        'Cholesterol_HDL': [chol_hdl], 'Cholesterol_LDL': [chol_ldl], 'GGT': [ggt],
        'Serum_Urate': [urate], 'Physical_Activity_Level': [activity],
        'Dietary_Intake_Calories': [calories], 'Alcohol_Consumption': [alcohol],
        'Smoking_Status': [smoking], 'Family_History_of_Diabetes': [family],
        'Previous_Gestational_Diabetes': [gestational]
    })
    return df_input

input_df = user_input()

# ----------------- Encode Categorical Variables -----------------
manual_mapping = {"Yes": 1, "No": 0, "Low": 0, "Medium": 1, "High": 2, "Male": 1, "Female": 0}

for col in input_df.columns:
    if input_df[col].dtype == 'object':
        if col in label_encoders:  # use saved encoder if available
            input_df[col] = label_encoders[col].transform(input_df[col])
        else:  # fallback manual mapping
            input_df[col] = input_df[col].map(manual_mapping).fillna(0).astype(int)

# ----------------- Align Features with Model -----------------
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ----------------- Prediction -----------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    st.subheader(" Prediction Result")
    if prediction == 1:
        st.error(f"Patient is likely to have diabetes.\n\n**Probability:** {prob:.2f}%")
    else:
        st.success(f"Patient is not likely to have diabetes.\n\n**Probability:** {prob:.2f}%")

st.markdown("---")
st.caption("Developed with using Streamlit & Random Forest Classifier.")
