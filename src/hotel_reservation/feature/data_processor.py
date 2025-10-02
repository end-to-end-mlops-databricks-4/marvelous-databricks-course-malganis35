"""Module for preprocessing hotel reservation data."""

import datetime
import time

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hotel_reservation.utils.config import ProjectConfig
from hotel_reservation.utils.timer import timeit


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize the DataProcessor.

        :param pandas_df: A pandas DataFrame containing the raw input data.
        :param config: A ProjectConfig object with parameters and table names.
        :param spark: An active SparkSession.
        """
        self.df = pandas_df
        self.config = config
        self.spark = spark

    @timeit
    def preprocess(self) -> pd.DataFrame:
        """Apply full preprocessing pipeline to the dataset.

        :return: A cleaned and feature-engineered pandas DataFrame.
        """
        self._drop_unused_columns()
        # self._create_features()
        # self._log_and_scale_numeric()
        self._drop_dupplicates()

        return self.df

    def _drop_unused_columns(self) -> None:
        """Drop unused or unnecessary columns from the DataFrame.

        Specifically removes 'Booking_ID' if present.
        Removes: 'arrival_date'
        """
        # self.df.drop(columns=["Booking_ID"], errors="ignore", inplace=True)
        self.df.drop(columns=["arrival_date"], errors="ignore", inplace=True)

    def _create_features(self) -> None:
        """Create engineered features in the DataFrame.

        Includes:
        - total_nights: sum of weekend and week nights
        - has_children: binary indicator if children are present
        - arrival_date_complete: datetime composed of year, month, and day
        """
        self.df["total_nights"] = self.df["no_of_weekend_nights"] + self.df["no_of_week_nights"]
        self.df["has_children"] = self.df["no_of_children"].apply(lambda x: 1 if x > 0 else 0)

    def _log_and_scale_numeric(self) -> None:
        """Apply log transformation and standard scaling to numeric features.

        - Applies log1p to 'lead_time' and 'avg_price_per_room' if they exist.
        - Scales available numerical columns to mean=0 and std=1.
        """
        for col in ["lead_time", "avg_price_per_room"]:
            if col in self.df.columns:
                self.df[col] = np.log1p(self.df[col])
        numerical_cols = [
            col
            for col in ["lead_time", "avg_price_per_room", "total_nights", "no_of_special_requests"]
            if col in self.df.columns
        ]
        scaler = StandardScaler()
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])

    def _drop_dupplicates(self) -> None:
        """Drop optional dupplicates data.

        Applies drop dupplicates from pandas.
        """
        self.df = self.df.drop_duplicates()

    @timeit
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        logger.info(f"Training set shape: {train_set.shape}")
        logger.info(f"Test set shape: {test_set.shape}")

        return train_set, test_set

    @timeit
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks Delta tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.train_table}"
        )

        logger.info(
            f"Train set saved to {self.config.catalog_name}.{self.config.schema_name}.{self.config.train_table}"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.test_table}"
        )

        logger.info(f"Test set saved to {self.config.catalog_name}.{self.config.schema_name}.{self.config.test_table}")

    @timeit
    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed (CDF) on the Delta tables for train and test sets.

        This method runs ALTER TABLE commands to activate delta.enableChangeDataFeed=true
        on both train and test tables in the Databricks catalog.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{self.config.train_table} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.{self.config.test_table} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

@timeit
def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 50) -> pd.DataFrame:
    """Generate synthetic data matching input DataFrame distributions with optional drift.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types. Supports intentional data drift for specific features when enabled.

    :param df: Source DataFrame containing original data distributions
    :param drift: Flag to activate synthetic data drift injection
    :param num_rows: Number of synthetic records to generate
    :return: DataFrame containing generated synthetic data
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Booking_ID":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"arrival_year"}:
                synthetic_data[column] = np.random.randint(df[column].min(), df[column].max() + 1, num_rows)
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

                if column in {  
                    "arrival_month",
                    "arrival_date",
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
                }:
                    synthetic_data[column] = np.maximum(0, synthetic_data[column])  # Ensure non-negative

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].dropna().unique(),
                num_rows,
                p=df[column].value_counts(normalize=True).reindex(df[column].dropna().unique(), fill_value=0).values,
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column].dropna(), num_rows)

    # Type alignment with Databricks schema
    int_columns = {
        "arrival_date",
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
    }

    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].round().astype(np.int32)  # ✅ int32 (Spark IntegerType)

    # Floats
    if "avg_price_per_room" in synthetic_data.columns:
        synthetic_data["avg_price_per_room"] = (
            pd.to_numeric(synthetic_data["avg_price_per_room"], errors="coerce").astype(np.float64)
        )

    # Strings
    string_columns = [
        "Booking_ID",
        "type_of_meal_plan",
        "room_type_reserved",
        "market_segment_type",
        "booking_status",
    ]
    for col in string_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].astype(str)

    # Add timestamp column (optional for later merge)
    # timestamp_base = int(time.time() * 1000)
    # synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        # Introduce drift in selected features
        for feature in ["avg_price_per_room", "lead_time"]:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 2

        # Adjust arrival_year drift
        current_year = pd.Timestamp.now().year
        if "arrival_year" in synthetic_data.columns:
            synthetic_data["arrival_year"] = np.random.randint(current_year - 2, current_year + 1, num_rows).astype(np.int32)

    return synthetic_data


@timeit
def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching Databricks schema with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
