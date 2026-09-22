
import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊"
)

# App heading
st.title("📊 Employee Performance Predictor")

st.write(
    "Enter Training Hours and Attendance to predict employee performance."
)

# Load ANN model
model = joblib.load("employee_performance_ann.pkl")

# Load scaler
scaler = joblib.load("employee_performance_scaler.pkl")

# Training Hours input
training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=30.0,
    value=8.0,
    step=1.0
)

# Attendance input
attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# Prediction button
if st.button("Predict Performance"):

    # Prepare input
    input_data = np.array([
        [training_hours, attendance]
    ])

    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # ANN prediction
    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]

    # Convert probability into result
    if probability >= 0.5:
        result = "Good"
    else:
        result = "Needs Improvement"

    # Display result
    st.subheader("Prediction Result")

    if result == "Good":
        st.success("Performance: GOOD")
    else:
        st.warning("Performance: NEEDS IMPROVEMENT")

    st.write(
        "Good Probability:",
        round(float(probability) * 100, 2),
        "%"
    )

    st.write(
        "Training Hours:",
        training_hours
    )

    st.write(
        "Attendance:",
        attendance,
        "%"
    )
