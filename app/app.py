import json
import os
import subprocess
import time
from typing import Any

import pandas as pd
import plotly.express as px
import pretty_errors  # noqa: F401
import requests
import streamlit as st
from databricks import sql
from dotenv import load_dotenv
from loguru import logger
from requests.auth import HTTPBasicAuth

# --- USER CONFIG ---
serving_endpoint = "https://dbc-f122dc18-1b68.cloud.databricks.com/serving-endpoints/hotel-reservation-basic-model-serving-db/invocations"

# --- FUNCTIONS ---


def set_page_config() -> None:
    """Set the Page Configuration."""
    logger.info("Configure page layout")
    st.set_page_config(page_title="Hotel Reservation Predictor", page_icon="🏨", layout="wide")


def set_app_config() -> None:
    """Set the App Configuration."""
    st.title("🔮 Hotel Reservation Classification (Databricks Unity Catalog Model)")
    st.text("Author: Cao Tri DO (alias malganis35)")
    st.markdown(
        """
        *This application showcases An end-to-end MLOps project developed as part of the *Marvelous MLOps Databricks Course (Cohort 4). It automates the complete lifecycle of a **hotel reservation classification model**, from **data ingestion & preprocessing** to **model training, registration, deployment, and serving** — fully orchestrated on **Databricks**. Start by making prediction in this interface*. It provides an easy way to monitor the hotel reservation ML System: 1/ **Generic monitoring** (system health, errors, latency); 2/ **ML Specific monitoring** (DQ, Data Drift); 3/ **Cost & Business Value** (Infra, Business Value, KPI); 4/ **Fairness & Bias**

        *The data are based on the Delta table: `mlops_dev.caotrido.model_monitoring`*
        """
    )


def get_token(DATABRICKS_HOST: str) -> str:
    """Retrieve an OAuth token from the Databricks workspace."""
    response = requests.post(
        f"{DATABRICKS_HOST}/oidc/v1/token",
        auth=HTTPBasicAuth(os.environ["DATABRICKS_CLIENT_ID"], os.environ["DATABRICKS_CLIENT_SECRET"]),
        data={"grant_type": "client_credentials", "scope": "all-apis"},
    )
    return response.json()["access_token"]


def get_databricks_token(DATABRICKS_HOST: str) -> str:
    """Automatically generates a Databricks temporary token via CLI.

    Args:
        DATABRICKS_HOST (str): The host URL of the Databricks instance.

    Returns:
        str: The JSON data containing the generated Databricks token.

    """
    logger.info("🔑 Automatically generating a Databricks temporary token via CLI...")

    result = subprocess.run(
        ["databricks", "auth", "token", "--host", DATABRICKS_HOST, "--output", "JSON"],
        capture_output=True,
        text=True,
        check=True,
    )

    token_data = json.loads(result.stdout)

    logger.info(f"✅ Temporary token acquired (expires at {token_data['expiry']})")

    return token_data


# Endpoint call function
def call_endpoint(record: list[dict[str, Any]], serving_endpoint: str) -> tuple[int, str]:
    """Call the Databricks model serving endpoint with a given input record."""
    logger.debug(f"Calling the endpoint url: {serving_endpoint}")

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


# --- MODEL CONFIGURATION ---
# Update this path to match your Unity Catalog setup

try:
    # Ensure host is prefixed properly
    raw_host = os.environ["DATABRICKS_HOST"]
    DATABRICKS_HOST = raw_host if raw_host.startswith("https://") else f"https://{raw_host}"
    db_token = get_token(DATABRICKS_HOST)

except Exception as e:
    logger.warning(f"Coding might be running locally. Returning: {e}")
    logger.debug(
        "Getting a token using the local .env file or requesting a temporary token if no token defined in .env file"
    )
    ENV_FILE = "./.env"
    load_dotenv(dotenv_path=ENV_FILE, override=True)
    profile = os.getenv("PROFILE")  # os.environ["PROFILE"]
    logger.debug(f"Detected profile: {profile}")
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

os.environ["DBR_TOKEN"] = db_token
os.environ["DATABRICKS_TOKEN"] = db_token  # required by Databricks SDK / Connect
os.environ["DBR_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST

DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_HOST").replace("https://", "")
DATABRICKS_HTTP_PATH = "/sql/1.0/warehouses/711fa33d05cc334c"  # os.getenv("DATABRICKS_WAREHOUSE_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")


# --- CONNEXION AU SQL WAREHOUSE ---
@st.cache_data(ttl=300)
def run_query(query: str) -> pd.DataFrame:
    """Exécuter une requête SQL sur Databricks."""
    with (
        sql.connect(
            server_hostname=DATABRICKS_SERVER_HOSTNAME,
            http_path=DATABRICKS_HTTP_PATH,
            access_token=DATABRICKS_TOKEN,
        ) as connection,
        connection.cursor() as cursor,
    ):
        cursor.execute(query)
        result = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return pd.DataFrame(result, columns=cols)


# --- STREAMLIT CONFIG ---
set_page_config()
set_app_config()


# --- SIDEBAR ---
with st.sidebar:
    st.title("🏨 Hotel Reservation Predictor")
    try:
        st.image("./hotel.png", width=300)
    except Exception as e:
        logger.warning(f"Coding might be running locally. Returning: {e}")
        st.image("./app/hotel.png", width=300)
    st.markdown(
        "This app predicts whether a hotel booking will be **honored or canceled** using a Databricks UC model."
    )
    st.markdown("**Instructions:**\n- Fill in booking details below\n- Click **Predict** to see the outcome")

# ======================================================
# 🧭 ONGLET NAVIGATION PRINCIPALE
# ======================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧭 Prediction Demo",
        "🧠 Generic Monitoring",
        "📊 ML Monitoring",
        "💰 Costs & Business Value",
        "⚖️ Fairness & Bias",
    ]
)

# 💅 Personnalisation du style des onglets
st.markdown(
    """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: 600;
    }
    button[data-baseweb="tab"] {
        padding: 0.8em 1.2em;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] p {
        color: #0072ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# 🧭 PREDICTION
# ======================================================

with tab1:
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
        avg_price_per_room = st.number_input(
            "Average Price per Room", min_value=20.0, max_value=500.0, value=100.0, step=5.0
        )
        no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=10, value=0)
        no_of_previous_bookings_not_canceled = st.number_input(
            "Previous Bookings (Not Canceled)", min_value=0, max_value=10, value=1
        )
        no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=1)

    # --- PREPARE DATAFRAME ---
    input_df = pd.DataFrame(
        [
            [
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
            ]
        ],
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

        dataframe_records = input_df.to_dict(orient="records")

        logger.debug("Making the prediction ...")
        # prediction = model.predict(input_df)

        # Test with one sample
        # 🕒 Spinner during processing
        with st.spinner("⏳ The model endpoint is starting... Please wait a few seconds..."):
            start_time = time.time()
            status_code, response_text = call_endpoint(dataframe_records, serving_endpoint)
            duration = time.time() - start_time

        logger.debug(f"Response Status: {status_code}")
        logger.debug(f"Response Text: {response_text}")

        # Parse the JSON returned by the endpoint
        try:
            response_json = json.loads(response_text)
            prediction_label = response_json.get("predictions", ["Unknown"])[0]
            outcome_tmp = f"✅ Prediction completed: {prediction_label}"
        except Exception as e:
            logger.error(f"Error parsing model response: {e}")
            outcome_tmp = f"❌ Error in prediction response: {response_text}"

        outcome = "✅ Booking likely honored" if prediction_label == "Not_Canceled" else "❌ Booking likely canceled"

        st.subheader(outcome)

        st.markdown("---")

        st.write("### 🔠 Input summary")
        st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str))
        logger.success(f"✅ Prediction completed: {outcome}")


# ======================================================
#   3️⃣  GENERIC MONITORING (System Health, Errors, Latency)
# ======================================================

with tab2:
    st.subheader("🧠 Generic Monitoring — System Health & Performance")

    st.markdown(
        """
        This table provides a technical overview of the prediction system:

        - **Health**: system availability (proportion of successful calls)
        - **Errors**: number of failed or invalid calls
        - **Latency**: average execution time of inferences
        """
    )

    # Vérifie que la colonne execution_duration_ms existe
    columns = run_query("SELECT * FROM mlops_dev.caotrido.model_monitoring LIMIT 1").columns
    has_latency = "execution_duration_ms" in columns

    # Exemple simplifié : calcul de la santé globale
    query_system = f"""
    SELECT
    date_trunc('hour', CAST(timestamp AS timestamp)) AS heure,
    COUNT(*) AS total_calls,
    SUM(CASE WHEN prediction IS NULL THEN 1 ELSE 0 END) AS errors,
    {"AVG(execution_duration_ms)" if has_latency else "NULL"} AS avg_latency_ms
    FROM mlops_dev.caotrido.model_monitoring
    WHERE CAST(timestamp AS timestamp) BETWEEN current_date() - INTERVAL 7 DAY AND current_timestamp()
    GROUP BY 1
    ORDER BY 1
    """

    df_system = run_query(query_system)

    query_calls = """
    SELECT
    date_trunc('hour', CAST(timestamp AS timestamp)) AS heure,
    COUNT(*) AS nombre_appels
    FROM mlops_dev.caotrido.model_monitoring
    WHERE CAST(timestamp AS timestamp) BETWEEN current_date() - INTERVAL 7 DAY AND current_timestamp()
    GROUP BY 1
    ORDER BY 1;
    """
    df_calls = run_query(query_calls)

    if df_system.empty:
        st.warning("⚠️ No data available for the monitoring system")
    else:
        df_system["error_rate"] = (df_system["errors"] / df_system["total_calls"]) * 100
        if has_latency:
            df_system["avg_latency_ms"] = df_system["avg_latency_ms"].astype(float)

        # ✅ System health indicator
        latest = df_system.iloc[-1]
        success_rate = 100 - latest["error_rate"]

        st.markdown("### 💡 System Health")
        _, c1, c2, c3 = st.columns([1, 2, 2, 2])
        c1.metric("✅ Health", f"{success_rate:.2f}%")
        c2.metric("⚠️ Errors", f"{int(latest['errors'])}")
        if has_latency:
            c3.metric("⏱️ Avg Latency (ms)", f"{latest['avg_latency_ms']:.1f}")
        else:
            c3.metric("⏱️ Avg Latency (ms)", "n/a")

        # 🔧 Line charts
        with st.expander("📈 Temporal Drilldown"):
            st.dataframe(df_system, width="stretch", hide_index=True)
            fig_health = px.line(df_system, x="heure", y="error_rate", title="Error rates (%) in time", markers=True)
            st.plotly_chart(fig_health, config={"displayModeBar": True, "responsive": True})

            if has_latency:
                fig_latency = px.line(
                    df_system, x="heure", y="avg_latency_ms", title="Mean Latency (ms) in time", markers=True
                )
                st.plotly_chart(fig_latency, config={"displayModeBar": True, "responsive": True})

    # --- METRICS SUMMARY ---
    st.subheader("📈 Nb Calls (inference)")

    if not df_calls.empty:
        total_calls = int(df_calls["nombre_appels"].sum())
        last_hour = int(df_calls["nombre_appels"].iloc[-1])

        _, c1, c2 = st.columns([1, 2, 2])
        c1.metric("📊 Total Calls", f"{total_calls}")
        c2.metric("🕒 Last hour calls", f"{last_hour}")

    if df_calls.empty:
        st.warning("⚠️ No data found in the **model_monitoring** table (last 7 days).")
    else:
        fig_calls = px.line(
            df_calls,
            x="heure",
            y="nombre_appels",
            title="Total Calls (inference)",
            markers=True,
        )
        st.plotly_chart(fig_calls, config={"displayModeBar": True, "responsive": True})

        with st.expander("🧭 How to interpretate the graph ?"):
            st.markdown(
                """
                ℹ️ **Interpretation:**
                This chart shows the frequency of calls to your Databricks prediction endpoint over time.

                * A sudden increase may indicate a **spike in traffic** or a **batch of inferences**.
                * A prolonged absence of calls may signal a **service outage** or a **drop in demand**.

                """
            )

# --- SECTION 2 : DATA DRIFT ---
with tab3:
    st.subheader("📊 ML Monitoring - Global Data Drift")

    # ======================================================
    #   1️⃣  VARIABLES NUMÉRIQUES
    # ======================================================
    st.markdown("### 🔢 Drift on numeric variables")

    num_features = [
        "arrival_month",
        "arrival_year",
        "avg_price_per_room",
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

    # Vérifie les colonnes existantes
    existing_columns = run_query("SELECT * FROM mlops_dev.caotrido.model_monitoring LIMIT 1").columns

    # Garde seulement les features présentes
    valid_num_features = [f for f in num_features if f in existing_columns]

    selected_num_features = st.multiselect(
        "🧮 Select the numerical variables to analyze:",
        options=valid_num_features,
        default=["avg_price_per_room", "lead_time", "no_of_children"],
        help="Select one or more numerical variables to calculate their hourly drift.",
    )

    if not selected_num_features:
        st.info("👉 Select at least one numerical variable to display the drift table.")
    else:
        drift_queries = []
        for feat in selected_num_features:
            drift_queries.append(f"""
            SELECT
                date_trunc('hour', CAST(timestamp AS timestamp)) AS heure,
                '{feat}' AS feature,
                ABS(AVG({feat}) - (SELECT AVG({feat}) FROM mlops_dev.caotrido.model_monitoring)) /
                (SELECT NULLIF(STDDEV({feat}), 0) FROM mlops_dev.caotrido.model_monitoring) AS drift_score
            FROM mlops_dev.caotrido.model_monitoring
            WHERE {feat} IS NOT NULL
            GROUP BY 1
            """)
        query_drift_num = " UNION ALL ".join(drift_queries) + " ORDER BY heure"

        df_drift_num = run_query(query_drift_num)

        if df_drift_num.empty:
            st.warning("⚠️ Not enough data to calculate the drift of numerical variables.")
        else:
            st.dataframe(df_drift_num.sort_values(["heure", "feature"]), width="stretch", hide_index=True)

            # Résumé statistique global
            mean_drift = df_drift_num["drift_score"].mean()
            max_drift = df_drift_num["drift_score"].max()
            st.markdown(
                f"""
                **📈 Average on drift score:** `{mean_drift:.3f}`
                **🚨 Maximal Drift score observed:** `{max_drift:.3f}`
                """
            )

            with st.expander("🧭 How to interpretate numerical drift ?"):
                st.markdown(
                    """
                    ℹ️ **Interpretation:**
                    - A `drift_score` ≈ **0** means the variable’s distribution is stable.
                    - A `drift_score` > **1** means the hourly average deviates significantly from the historical mean.
                    This may indicate:
                    - a **seasonality change**,
                    - a **change in user behavior**,
                    - or a **data collection anomaly**.
                    """
                )

    # ======================================================
    #   2️⃣  VARIABLES CATÉGORIELLES
    # ======================================================
    st.markdown("### 🔠 Drift on categorical variables")

    cat_features = [
        "market_segment_type",
        "room_type_reserved",
        "type_of_meal_plan",
    ]

    selected_cat_features = st.multiselect(
        "🔍 Select the categorical variables to analyze:",
        options=cat_features,
        default=["market_segment_type"],
        help="Select one or more categorical variables to calculate their hourly drift.",
    )

    if not selected_cat_features:
        st.info("👉 Select at least one categorical variable to display the drift table.")
    else:
        all_results = []  # contiendra les DataFrames concaténés

        for feat in selected_cat_features:
            query = f"""
            WITH base AS (
                SELECT
                    date_trunc('hour', CAST(timestamp AS timestamp)) AS heure,
                    {feat} AS category
                FROM mlops_dev.caotrido.model_monitoring
                WHERE {feat} IS NOT NULL
            ),
            global_dist AS (
                SELECT
                    category,
                    COUNT(*) / SUM(COUNT(*)) OVER() AS global_freq
                FROM base
                GROUP BY category
            ),
            hourly_dist AS (
                SELECT
                    heure,
                    category,
                    COUNT(*) / SUM(COUNT(*)) OVER(PARTITION BY heure) AS hourly_freq
                FROM base
                GROUP BY heure, category
            )
            SELECT
                h.heure,
                '{feat}' AS feature,
                SUM(ABS(h.hourly_freq - g.global_freq)) AS drift_score
            FROM hourly_dist h
            JOIN global_dist g
            ON h.category = g.category
            GROUP BY h.heure
            ORDER BY h.heure
            """

            try:
                df_temp = run_query(query)
                if not df_temp.empty:
                    all_results.append(df_temp)
            except Exception as e:
                st.error(f"Erreur lors du calcul de la dérive pour `{feat}` : {e}")

        # Fusionner tous les résultats dans un seul tableau
        if not all_results:
            st.warning("⚠️ Not enough data to calculate the drift of categorical variables.")
        else:
            df_drift_cat = pd.concat(all_results, ignore_index=True)

            st.dataframe(df_drift_cat.sort_values(["heure", "feature"]), width="stretch", hide_index=True)

            # Résumé statistique global
            mean_drift = df_drift_cat["drift_score"].mean()
            max_drift = df_drift_cat["drift_score"].max()
            st.markdown(
                f"""
                **📈 Average on drift score:** `{mean_drift:.3f}`
                **🚨 Maximal Drift score observed:** `{max_drift:.3f}`
                """
            )

            with st.expander("🧭 How to interpretate categorical drift ?"):
                st.markdown(
                    """
                    ℹ️ **Interpretation:**
                    - A `drift_score` ≈ **0** means the class distribution remains stable.
                    - A high `drift_score` means certain categories appear or disappear more frequently.
                    This may indicate:
                    - a **change in user profile**,
                    - a **modification in data sources**,
                    - or a **sampling bias**.
                    """
                )

    # ======================================================
    #   4️⃣  ML-SPECIFIC MONITORING — Data Quality Checks
    # ======================================================
    st.markdown("---")
    st.subheader("🤖 Data Quality Checks")

    st.markdown(
        """
        This section checks the **data quality** received by the model (Missing value rate, Presence of anomalies (simple outliers), Statistical stability of distributions)
        """
    )

    quality_features = [
        "arrival_month",
        "arrival_year",
        "avg_price_per_room",
        "lead_time",
        "no_of_adults",
        "no_of_children",
        "no_of_special_requests",
    ]

    selected_quality_features = st.multiselect(
        "📋 Select the variables to include in the quality check.:",
        options=quality_features,
        default=["avg_price_per_room", "lead_time", "no_of_children"],
        help="Choose the variables for which you want to assess data quality.",
    )

    if not selected_quality_features:
        st.info("👉 Select at least one variable to start the quality check.")
    else:
        subqueries = []
        for feat in selected_quality_features:
            subqueries.append(f"""
            WITH stats AS (
                SELECT
                    AVG({feat}) AS mean_val,
                    STDDEV({feat}) AS std_val
                FROM mlops_dev.caotrido.model_monitoring
                WHERE {feat} IS NOT NULL
            )
            SELECT
                '{feat}' AS feature,
                COUNT(*) AS total_rows,
                SUM(CASE WHEN {feat} IS NULL THEN 1 ELSE 0 END) AS missing_count,
                AVG(CASE WHEN {feat} IS NULL THEN 1.0 ELSE 0.0 END) AS missing_rate,
                AVG(
                    CASE
                        WHEN {feat} IS NOT NULL AND (
                            {feat} < s.mean_val - 3 * s.std_val OR
                            {feat} > s.mean_val + 3 * s.std_val
                        ) THEN 1.0 ELSE 0.0
                    END
                ) AS outlier_rate
            FROM mlops_dev.caotrido.model_monitoring
            CROSS JOIN stats s
            """)
        # Pour Databricks, chaque requête WITH doit être indépendante :
        # donc on exécute séparément et concatène les DataFrames Python
        df_list = []
        for q in subqueries:
            df_part = run_query(q)
            df_list.append(df_part)

        df_quality = pd.concat(df_list, ignore_index=True)

        if df_quality.empty:
            st.warning("⚠️ Not enough data to assess data quality.")
        else:
            df_quality["missing_rate"] = df_quality["missing_rate"].astype(float) * 100
            df_quality["outlier_rate"] = df_quality["outlier_rate"].astype(float) * 100

            # ✅ Tableau résumé
            st.dataframe(df_quality[["feature", "missing_rate", "outlier_rate"]], width="stretch", hide_index=True)

            # ✅ Graphique synthétique
            with st.expander("📈 Visualization of the data quality"):
                fig_missing = px.bar(
                    df_quality,
                    x="feature",
                    y="missing_rate",
                    text_auto=".2f",
                    title="Missing value ratio (%)",
                )
                fig_missing.update_traces(marker_color="orange")
                fig_missing.update_layout(yaxis_title="Ratio (%)", xaxis_title="Variable", showlegend=False, height=400)
                st.plotly_chart(fig_missing, config={"displayModeBar": True, "responsive": True})

                fig_outliers = px.bar(
                    df_quality,
                    x="feature",
                    y="outlier_rate",
                    text_auto=".2f",
                    title="Anomaly ratio (outliers) (%)",
                )
                fig_outliers.update_traces(marker_color="blue")
                fig_outliers.update_layout(
                    yaxis_title="Ratio (%)", xaxis_title="Variable", showlegend=False, height=400
                )
                st.plotly_chart(fig_outliers, config={"displayModeBar": True, "responsive": True})

            with st.expander("🧭 How to interpretate these metrics ?"):
                st.markdown(
                    """
                    ℹ️ **Interpretation:**
                    - A `missing_rate` > **5%** may indicate a data collection issue.
                    - An `outlier_rate` > **1%** suggests atypical or incorrectly entered data.
                    - These anomalies can directly impact the model’s stability and performance.
                    """
                )


# ======================================================
#   6️⃣  COSTS & BUSINESS VALUE MONITORING (INTERACTIVE)
# ======================================================
with tab4:
    st.subheader("💰 Costs & Business Value Monitoring")

    st.markdown(
        """
        This section links the **technical indicators** to the **business value** of the model:

        * 💵 *Infrastructure costs* related to inferences
        * 📈 *Business value* generated by predictions
        * 🎯 *Adjustable economic KPIs*
        """
    )

    # ======================================================
    # 🧮 Paramètres métier ajustables par l'utilisateur
    # ======================================================

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Business Parameters")

    COST_PER_INFERENCE = st.sidebar.number_input(
        "💸 Cost by inference (€)",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.001,
        help="Unit cost charged for each model call (Databricks, API, etc.).",
    )

    VALUE_PER_SUCCESSFUL_BOOKING = st.sidebar.number_input(
        "🏨 Average value of a reservation (€).)",
        min_value=0,
        max_value=1000,
        value=120,
        step=10,
        help="Average business value of a kept (honored) reservation.",
    )

    LOSS_PER_CANCELLATION = st.sidebar.number_input(
        "❌ Average loss per cancellation (€).",
        min_value=0,
        max_value=500,
        value=30,
        step=5,
        help="Estimated loss when a reservation is canceled.",
    )

    # ======================================================
    # 📊 Requête SQL — Récupération des données de prédiction
    # ======================================================
    query_business = """
    SELECT
    date_trunc('day', CAST(timestamp AS timestamp)) AS jour,
    COUNT(*) AS total_calls,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS canceled_pred,
    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS kept_pred
    FROM mlops_dev.caotrido.model_monitoring
    WHERE CAST(timestamp AS timestamp) BETWEEN current_date() - INTERVAL 7 DAY AND current_timestamp()
    GROUP BY 1
    ORDER BY 1
    """

    df_business = run_query(query_business)

    # ======================================================
    # 💹 Calculs économiques
    # ======================================================
    if df_business.empty:
        st.warning("⚠️ Not enough data to estimate costs and business value.")
    else:
        df_business["infra_cost"] = df_business["total_calls"] * COST_PER_INFERENCE
        df_business["estimated_value"] = (
            df_business["kept_pred"] * VALUE_PER_SUCCESSFUL_BOOKING
            - df_business["canceled_pred"] * LOSS_PER_CANCELLATION
            - df_business["infra_cost"]
        )

        total_cost = df_business["infra_cost"].sum()
        total_value = df_business["estimated_value"].sum()
        total_calls = int(df_business["total_calls"].sum())
        avg_value_per_call = total_value / total_calls if total_calls > 0 else 0

        # ======================================================
        # 🎯 Indicateurs clés (KPIs)
        # ======================================================
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("💸 Total Cost", f"{total_cost:,.2f} €")
        kpi2.metric("💰 Estimated generated value", f"{total_value:,.2f} €")
        kpi3.metric("📊 Average value / call", f"{avg_value_per_call:,.2f} €")

        # ======================================================
        # 📈 Graphiques
        # ======================================================

        with st.expander("📈 Evolution of costs and generated value."):
            fig_costs = px.line(
                df_business,
                x="jour",
                y=["infra_cost", "estimated_value"],
                title="Evolution of infrastructure costs and business value (last 7 days).",
                markers=True,
                labels={"value": "Estimated Value (€)", "jour": "Day", "variable": "Type"},
            )
            st.plotly_chart(fig_costs, config={"displayModeBar": True, "responsive": True})

        with st.expander("📊 Distribution of predictions."):
            fig_pred = px.bar(
                df_business,
                x="jour",
                y=["kept_pred", "canceled_pred"],
                barmode="stack",
                title="Volume des prédictions par jour",
                labels={"value": "Number of Predictions", "jour": "Day", "variable": "Type"},
            )
            st.plotly_chart(fig_pred, config={"displayModeBar": True, "responsive": True})

        # ======================================================
        # 🧭 Interprétation
        # ======================================================
        with st.expander("🧭 How to interpretate these KPIs ?"):
            st.markdown(
                """
                ℹ️ **Interpretation:**
                - The inference cost depends on the number of model calls.
                - The generated value is an estimate based on the predictions.
                - The parameters on the side allow you to test different economic scenarios:
                - 💸 unit cost of the model (cloud, API, GPU)
                - 💰 value of a reservation
                - ❌ cost of a cancellation

                🎯 Objective: **maximize the net value** (*generated value − infrastructure cost*).
                """
            )


# ======================================================
#   5️⃣  FAIRNESS & BIAS MONITORING
# ======================================================
with tab5:
    st.subheader("⚖️ Fairness & Bias Monitoring")

    st.markdown(
        """
        This section evaluates the **fairness** of the model, meaning its **equity across groups**.
        It helps identify whether certain segments (e.g., customer types, meal plans, room types)
        receive **positive predictions (cancellations)** more or less frequently.
        """
    )

    # --- Variables catégorielles disponibles pour la segmentation ---
    bias_features = [
        "market_segment_type",
        "room_type_reserved",
        "type_of_meal_plan",
    ]

    selected_bias_feature = st.selectbox(
        "🎯 Choose the sensitive variable to analyze:",
        options=bias_features,
        index=0,
        help="This variable will be used to segment the predictions and assess disparities.",
    )

    query_bias = f"""
    WITH stats AS (
        SELECT
            {selected_bias_feature} AS category,
            COUNT(*) AS total,
            SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS positive_preds
        FROM mlops_dev.caotrido.model_monitoring
        WHERE {selected_bias_feature} IS NOT NULL
        GROUP BY {selected_bias_feature}
    )
    SELECT
        category,
        total,
        positive_preds,
        ROUND(positive_preds / total, 4) AS positive_rate,
        ROUND(
            (positive_preds / total) - AVG(positive_preds / total) OVER(),
            4
        ) AS bias_gap
    FROM stats
    ORDER BY positive_rate DESC
    """

    df_bias = run_query(query_bias)

    if df_bias.empty:
        st.warning("⚠️ Not enough data to evaluate the model’s fairness.")
    else:
        # ✅ Tableau de fairness
        st.dataframe(df_bias, width="stretch", hide_index=True)

        # ✅ Graphique — Taux de prédictions positives par groupe
        with st.expander("📊 Visualization of prediction disparity."):
            fig_bias = px.bar(
                df_bias,
                x="category",
                y="positive_rate",
                text_auto=".2%",
                title=f"Rate of positive predictions by group — {selected_bias_feature}",
            )
            fig_bias.update_layout(yaxis_title="Positive prediction rate (%)", xaxis_title="Group", height=400)
            st.plotly_chart(fig_bias, config={"displayModeBar": True, "responsive": True})

        # ✅ Interprétation
        with st.expander("🧭 How to interpretate these results ?"):
            st.markdown(
                """
                ℹ️ **Interpretation:**
                - The `positive_rate` indicates the proportion of "canceled" predictions within each group.
                - The `bias_gap` measures the difference between this rate and the overall average:
                - close to **0** → model is fair across groups;
                - positive → the group is **more often predicted as canceled**;
                - negative → the group is **less often predicted as canceled**.

                ⚠️ **Note:**
                - Differences greater than **0.1 (10%)** may indicate a significant bias.
                - These analyses should be **cross-checked with actual data** before concluding bias.
                """
            )
