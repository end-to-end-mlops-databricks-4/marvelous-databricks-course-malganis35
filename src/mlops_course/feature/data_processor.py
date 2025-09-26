"""Module for preprocessing hotel reservation data."""

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlops_course.utils.config import ProjectConfig
from mlops_course.utils.timer import timeit


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
        self._create_features()
        self._log_and_scale_numeric()
        self._cleanup_columns()

        return self.df

    def _drop_unused_columns(self) -> None:
        """Drop unused or unnecessary columns from the DataFrame.

        Specifically removes 'Booking_ID' if present.
        """
        self.df.drop(columns=["Booking_ID"], errors="ignore", inplace=True)

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

    def _cleanup_columns(self) -> None:
        """Drop temporary or redundant date columns from the DataFrame.

        Removes: 'arrival_date'
        """
        self.df.drop(columns=["arrival_date"], errors="ignore", inplace=True)

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
