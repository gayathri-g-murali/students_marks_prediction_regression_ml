import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the trained model
model = joblib.load("best_model.pkl")

# App title
st.title("🎓 Student Exam Score Predictor")

st.markdown("Predict a student's exam performance based on lifestyle and academic inputs.")

# Input sliders
study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 2.0)
attendance = st.slider("📊 Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("🧠 Mental Health Rating (1–10)", 1, 10, 5)
sleep_hours = st.slider("💤 Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("💼 Part-Time Job", ["No", "Yes"])

# Encode Part-Time Job
ptj_encoded = 1 if part_time_job == "Yes" else 0

# Predict button
if st.button("🔍 Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # Clamp score between 0 and 100

    st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

    # Model information box
    st.info("ℹ️ This prediction is based on a machine learning model trained on academic and lifestyle data. "
            "The model achieved ~80% R² accuracy during testing.")

    # Contextual advice (optional)
    if prediction >= 85:
        st.info("✅ High potential! Keep maintaining your habits.")
    elif prediction >= 60:
        st.info("⚠️ Decent score, but there’s room to grow with better consistency.")
    else:
        st.info("🔄 You may benefit from improving study hours, sleep, or mental well-being.")

# Optional footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Model: Regression | Author: Gayathri G Murali ")