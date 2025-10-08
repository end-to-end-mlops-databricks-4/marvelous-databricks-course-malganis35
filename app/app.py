
import os
import mlflow
import pandas as pd
import requests
import streamlit as st
from mlflow.pyfunc import PyFuncModel
from requests.auth import HTTPBasicAuth

import pretty_errors  # noqa: F401
from dotenv import load_dotenv
from loguru import logger

from hotel_reservation.utils.databricks_utils import create_spark_session, get_databricks_token, is_databricks

# --- STREAMLIT CONFIG ---

def set_page_config():
    logger.info("Configure page layout")
    st.set_page_config(page_title="Hotel Reservation Predictor", 
                       page_icon="🏨", 
                       layout="wide")


def set_app_config():
    st.title("🔮 Hotel Reservation Classification (Databricks Unity Catalog Model)")
    st.text("Author: Cao Tri DO (alias malganis35)")
    st.markdown(
        "*This application showcases An end-to-end MLOps project developed as part of the *Marvelous MLOps Databricks Course (Cohort 4). It automates the complete lifecycle of a **hotel reservation classification model**, from **data ingestion & preprocessing** to **model training, registration, deployment, and serving** — fully orchestrated on **Databricks**. Start by making prediction in this interface*"
    )


set_page_config()
set_app_config()


# --- MODEL CONFIGURATION ---
# Update this path to match your Unity Catalog setup
MODEL_URI = "models:/mlops_dev.caotrido.hotel_reservation_lr@latest-model"

def get_token() -> str:
    """Retrieve an OAuth token from the Databricks workspace."""
    response = requests.post(
        f"{host}/oidc/v1/token",
        auth=HTTPBasicAuth(os.environ["DATABRICKS_CLIENT_ID"], os.environ["DATABRICKS_CLIENT_SECRET"]),
        data={"grant_type": "client_credentials", "scope": "all-apis"},
    )
    return response.json()["access_token"]

if is_databricks():
    # Ensure host is prefixed properly
    raw_host = os.environ["DATABRICKS_HOST"]
    host = raw_host if raw_host.startswith("https://") else f"https://{raw_host}"
    os.environ["DATABRICKS_HOST"] = host
    
    os.environ["DATABRICKS_TOKEN"] = get_token()
    
    mlflow.set_registry_uri("databricks-uc")
else:
    ENV_FILE = f"./.env"
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    logger.debug(profile)
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    
    # Generate a temporary Databricks access token using the CLI
    if os.getenv("DATABRICKS_TOKEN"):
        logger.debug("Existing databricks token in .env file")
        db_token = os.getenv("DATABRICKS_TOKEN")
    else:
        logger.debug("No databricks token in .env file. Getting a temporary token ...")
        token_data = get_databricks_token(DATABRICKS_HOST)
        db_token = token_data["access_token"]
        logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")
    
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


@st.cache_resource
def load_uc_model() -> PyFuncModel:
    """Load the model from Unity Catalog."""
    return mlflow.pyfunc.load_model(MODEL_URI)


model = load_uc_model()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 Hotel Reservation Predictor")
    st.image("./app/hotel.png", width=300)
    st.markdown("This app predicts whether a hotel booking will be **honored or canceled** using a Databricks UC model.")
    st.markdown("**Instructions:**\n- Fill in booking details below\n- Click **Predict** to see the outcome")

# --- INPUT LAYOUT ---
st.subheader("📋 Input informations for reservation")
col1, col2, col3 = st.columns(3)

with col1:
    no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    no_of_week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)
    type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])

with col2:
    required_car_parking_space = st.selectbox("Car Parking Space Required", [0, 1])
    room_type_reserved = st.selectbox(
        "Room Type Reserved",
        ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5"],
    )
    lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, max_value=365, value=30)
    arrival_year = st.number_input("Arrival Year", min_value=2020, max_value=2030, value=2025)
    arrival_month = st.selectbox("Arrival Month", list(range(1, 13)))

with col3:
    market_segment_type = st.selectbox(
        "Market Segment Type",
        ["Online", "Offline", "Corporate", "Complementary", "Aviation"],
    )
    repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
    avg_price_per_room = st.number_input("Average Price per Room", min_value=20.0, max_value=500.0, value=100.0, step=5.0)
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0)
    no_of_previous_bookings_not_canceled = st.number_input(
        "Previous Bookings (Not Canceled)", min_value=0, max_value=10, value=1
    )
    no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)

# --- PREPARE DATAFRAME ---
input_df = pd.DataFrame(
    [[
        no_of_adults,
        no_of_children,
        no_of_weekend_nights,
        no_of_week_nights,
        type_of_meal_plan,
        required_car_parking_space,
        room_type_reserved,
        lead_time,
        arrival_year,
        arrival_month,
        market_segment_type,
        repeated_guest,
        avg_price_per_room,
        no_of_previous_cancellations,
        no_of_previous_bookings_not_canceled,
        no_of_special_requests,
    ]],
    columns=[
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "market_segment_type",
        "repeated_guest",
        "avg_price_per_room",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "no_of_special_requests",
    ],
)

st.markdown("---")

# --- PREDICTION ---
st.subheader("🔎 Prediction for ML Model")
if st.button("🚀 Predict Booking Outcome"):
    logger.info("Asking for a predicition ...")
    logger.debug("Convert the type to align to Unity Catalog")
    int_columns = [
        "arrival_month",
        "arrival_year",
        "lead_time",
        "no_of_adults",
        "no_of_children",
        "no_of_previous_bookings_not_canceled",
        "no_of_previous_cancellations",
        "no_of_special_requests",
        "no_of_week_nights",
        "no_of_weekend_nights",
        "repeated_guest",
        "required_car_parking_space",
    ]
    float_columns = ["avg_price_per_room"]

    input_df[int_columns] = input_df[int_columns].astype("int32")
    input_df[float_columns] = input_df[float_columns].astype("float64")
    
    logger.debug("Making the prediction ...")
    prediction = model.predict(input_df)

    # Handling output for classification (0/1)
    outcome = "✅ Booking likely honored" if prediction[0] == 1 else "❌ Booking likely canceled"

    st.subheader(outcome)

    st.write("### Input summary")
    st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str))
    logger.success(f"Prediction completed: {outcome}")
