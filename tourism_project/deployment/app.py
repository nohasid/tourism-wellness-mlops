"""
Streamlit app for the "Visit with Us" Wellness Tourism Package predictor.

Deployed on Streamlit Community Cloud, which clones this GitHub repo
directly and runs this file. The trained model
(tourism_project/deployment/best_model.joblib) is committed into the repo
by the CI pipeline's model-training job -- see
.github/workflows/pipeline.yml -- and loaded here via a path relative to
this file's own location, so it works regardless of the process's current
working directory.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_ARTIFACT_FILENAME = "best_model.joblib"
MODEL_PATH = Path(__file__).parent / MODEL_ARTIFACT_FILENAME


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}. Make sure "
            "tourism_project/model_building/train.py has been run and its "
            "output committed to this repo before deploying."
        )
        st.stop()
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧘")
st.title("🧘 Wellness Tourism Package -- Purchase Predictor")
st.write(
    "Enter a customer's profile and sales-interaction details to predict "
    "whether they are likely to purchase the Wellness Tourism Package."
)

model = load_model()

with st.form("customer_form"):
    st.subheader("Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 36)
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
    with col2:
        monthly_income = st.number_input(
            "Monthly Income", min_value=1000, max_value=100000, value=22000, step=500
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
        own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
        passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
        num_children = st.slider("Number of Children Visiting (below age 5)", 0, 3, 0)

    st.subheader("Trip preferences")
    col3, col4 = st.columns(2)
    with col3:
        num_persons = st.slider("Number of Persons Visiting", 1, 5, 2)
        num_trips = st.slider("Average Number of Trips per Year", 1, 22, 3)
    with col4:
        preferred_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5], index=0)

    st.subheader("Sales interaction")
    col5, col6 = st.columns(2)
    with col5:
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox(
            "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
    with col6:
        duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
        num_followups = st.slider("Number of Follow-ups", 1, 6, 4)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": float(monthly_income),
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    st.divider()
    if prediction == 1:
        st.success(f"✅ Likely to purchase -- predicted probability: {probability:.1%}")
    else:
        st.error(f"❌ Unlikely to purchase -- predicted probability: {probability:.1%}")

    with st.expander("See the input sent to the model"):
        st.dataframe(input_df)
