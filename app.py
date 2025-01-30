import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = pickle.load(open(r"C:\Users\Ranjan kumar pradhan\.vscode\project_vs\Student_performance_predict\model.pkl", "rb"))

# Initialize LabelEncoders
internet_encoder = LabelEncoder()
parent_education_encoder = LabelEncoder()

# Assuming these were the original categories used during training
internet_encoder.classes_ = np.array(['No', 'Yes'])
parent_education_encoder.classes_ = np.array(['High School', 'College', 'Bachelor', 'Master'])

st.title("🎓 Student Performance Prediction API")

# User input fields
study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=12.0, value=3.0)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=75)
internet_access = st.selectbox("Internet Access at Home", ['No', 'Yes'])
parent_education = st.selectbox("Parent's Highest Education Level", ['High School', 'College', 'Bachelor', 'Master'])

# Encode categorical values
internet_encoded = internet_encoder.transform([internet_access])[0]
parent_education_encoded = parent_education_encoder.transform([parent_education])[0]

# Predict button
if st.button("Predict Final Score"):
    input_data = np.array([[study_hours, attendance, internet_encoded, parent_education_encoded]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")