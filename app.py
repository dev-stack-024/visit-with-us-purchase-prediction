import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# App title
st.title("Wellness Tourism Package Purchase Prediction")

st.write("Enter customer details to predict purchase likelihood")

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="krishnagunda/wellness-tourism-purchase-model",
    filename="wellness_rf_model.pkl"
)

model = joblib.load(model_path)

# Input fields
age = st.number_input("Age", min_value=18, max_value=100)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10)
preferred_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips", min_value=0, max_value=20)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
num_children = st.number_input("Number of Children", min_value=0, max_value=5)
designation = st.text_input("Designation")
monthly_income = st.number_input("Monthly Income", min_value=0)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
num_followups = st.number_input("Number of Followups", min_value=0)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1)

# Create dataframe
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase (Probability: {probability:.2f})")
    else:
        st.warning(f"Customer is unlikely to purchase (Probability: {probability:.2f})")