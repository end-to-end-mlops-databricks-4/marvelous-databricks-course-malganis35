"""Script to test predictions using the latest registered hotel_reservation model."""

import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Imports du projet
from hotel_reservation.model.custom_model import CustomModel
from hotel_reservation.utils.config import ProjectConfig, Tags
from hotel_reservation.utils.databricks_utils import create_spark_session, is_databricks

if __name__ == "__main__":
    # --- Setup général ---
    root_path = "."
    CONFIG_FILE = f"{root_path}/project_config.yml"
    ENV_FILE = f"{root_path}/.env"
    BRANCH = "dev"

    # --- Initialisation MLflow ---
    if not is_databricks():
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        profile = os.getenv("PROFILE")
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")

    logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow Registry URI: {mlflow.get_registry_uri()}")

    # --- Chargement de la config et du SparkSession ---
    config = ProjectConfig.from_yaml(config_path=CONFIG_FILE, env=BRANCH)
    spark = create_spark_session()

    # --- Création des tags ---
    tags_dict = {"git_sha": "none", "branch": BRANCH, "job_run_id": "manual_test"}
    tags = Tags(**tags_dict)

    # --- Instanciation du modèle ---
    custom_model = CustomModel(config=config, tags=tags, spark=spark)

    # --- Définir les colonnes attendues ---
    required_columns = [
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
        "market_segment_type",
        "room_type_reserved",
        "type_of_meal_plan",
    ]

    # --- Exemple de données à prédire ---
    example_data = pd.DataFrame(
        [
            {
                "arrival_month": 8,
                "arrival_year": 2024,
                "avg_price_per_room": 110.50,
                "lead_time": 45,
                "no_of_adults": 2,
                "no_of_children": 1,
                "no_of_previous_bookings_not_canceled": 0,
                "no_of_previous_cancellations": 0,
                "no_of_special_requests": 2,
                "no_of_week_nights": 5,
                "no_of_weekend_nights": 2,
                "repeated_guest": 0,
                "required_car_parking_space": 1,
                "market_segment_type": "Online",
                "room_type_reserved": "Deluxe",
                "type_of_meal_plan": "Meal Plan 1",
            },
            {
                "arrival_month": 3,
                "arrival_year": 2025,
                "avg_price_per_room": 80.75,
                "lead_time": 10,
                "no_of_adults": 1,
                "no_of_children": 0,
                "no_of_previous_bookings_not_canceled": 2,
                "no_of_previous_cancellations": 1,
                "no_of_special_requests": 0,
                "no_of_week_nights": 3,
                "no_of_weekend_nights": 0,
                "repeated_guest": 1,
                "required_car_parking_space": 0,
                "market_segment_type": "Corporate",
                "room_type_reserved": "Standard",
                "type_of_meal_plan": "Not Selected",
            },
        ]
    )

    # --- Harmonisation des données ---
    logger.info("🧾 Données brutes d'entrée :")
    logger.info(f"\n{example_data}")

    # Convertir tous les entiers 64 bits en int32 pour respecter le schéma MLflow
    for col in example_data.select_dtypes(include="int64").columns:
        example_data[col] = example_data[col].astype("int32")

    logger.info(f"✅ Types des colonnes après harmonisation :\n{example_data.dtypes}")

    # --- Prédiction ---
    logger.info("🚀 Chargement du modèle et exécution de la prédiction...")
    predictions_df = custom_model.load_latest_model_and_predict(example_data)

    # --- Résultats ---
    logger.info("✅ Prédictions terminées.")
    print("\n=== Résultats de la prédiction ===")
    print(predictions_df)
