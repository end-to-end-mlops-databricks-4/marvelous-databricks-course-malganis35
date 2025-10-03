"""Module for preprocessing hotel reservation data."""

import numpy as np
import pandas as pd
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
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
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, write_mode: str = "overwrite") -> None:
        """Save the train and test sets into Databricks Delta tables with mode control and CDF on first creation.

        :param train_set: pandas DataFrame (train)
        :param test_set: pandas DataFrame (test)
        :param write_mode: 'overwrite', 'append', or 'upsert'
        """

        def _write_with_mode(spark_df: DataFrame, table_name: str, merge_key: str = "Booking_ID") -> None:
            full_table_name = f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"

            # 1) Existence de la table
            table_exists = self._check_table_exists(full_table_name)

            # 2) Ajouter un timestamp technique
            df_with_ts = spark_df.withColumn("update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

            # Nb lines before writing to Unity Catalog
            previous_count = self._get_table_row_count(full_table_name) if table_exists else 0
            new_count = df_with_ts.count()

            logger.debug(f"📊 Table: {full_table_name}")
            logger.debug(f"    → Existing rows: {previous_count:,}")
            logger.debug(f"    → Incoming rows: {new_count:,}")

            # 3) Si la table n'existe pas → création + CDF
            if not table_exists:
                logger.warning(f"Table {full_table_name} not found. Creating it in 'overwrite' mode.")
                (
                    df_with_ts.write.mode("overwrite")
                    .option("mergeSchema", "true")  # utile si le schéma évolue
                    .saveAsTable(full_table_name)
                )
                logger.success(f"Table {full_table_name} created.")
                self._enable_change_data_feed(full_table_name)  # ✅ activer CDF à la création
                logger.info(f"CDF enabled for {full_table_name}.")
                return  # première exécution: on s'arrête ici

            # 4) Si la table existe → appliquer write_mode
            if write_mode in {"overwrite", "append"}:
                logger.info(f"Writing to {full_table_name} with mode '{write_mode}'")
                (df_with_ts.write.mode(write_mode).option("mergeSchema", "true").saveAsTable(full_table_name))
            elif write_mode == "upsert":
                logger.info(f"Performing UPSERT into {full_table_name} on key '{merge_key}'")
                if merge_key not in df_with_ts.columns:
                    raise ValueError(
                        f"UPSERT requested but merge key '{merge_key}' missing from data columns: {df_with_ts.columns}"
                    )
                self._merge_delta_table(df_with_ts, full_table_name, merge_key=merge_key)
            else:
                raise ValueError(f"Invalid write_mode: {write_mode}")

            logger.success(f"Data successfully written to {full_table_name} in mode '{write_mode}'.")

            # ✅ Check after writing to Unity Catalog
            final_count = self._get_table_row_count(full_table_name)
            diff = final_count - previous_count

            logger.debug("Checking consistency of the data after writting in Unity Catalog")
            logger.debug(f"    → Rows after write: {final_count:,}")
            logger.debug(f"    → Change: {diff:+,} rows")

            if final_count == previous_count and write_mode != "overwrite":
                logger.warning(f"⚠️ No new rows detected in {full_table_name}. Check your write mode or merge logic.")
            elif final_count < previous_count:
                logger.error(f"❌ Row count decreased in {full_table_name}! Possible data loss.")
            else:
                logger.success(f"✅ Write completed successfully for {full_table_name}.")

        # pandas → Spark
        train_spark_df = self.spark.createDataFrame(train_set)
        test_spark_df = self.spark.createDataFrame(test_set)

        _write_with_mode(train_spark_df, self.config.train_table, merge_key="Booking_ID")
        _write_with_mode(test_spark_df, self.config.test_table, merge_key="Booking_ID")

    def _check_table_exists(self, full_table_name: str) -> bool:
        """Return True if the fully qualified table exists in Unity Catalog."""
        try:
            self.spark.sql(f"DESCRIBE TABLE {full_table_name}")
            return True
        except Exception:
            return False

    def _enable_change_data_feed(self, full_table_name: str) -> None:
        """Enable Delta Change Data Feed for a table."""
        try:
            self.spark.sql(f"ALTER TABLE {full_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
            logger.success(f"CDF enabled for {full_table_name}")
        except Exception as e:
            logger.error(f"Failed to enable CDF on {full_table_name}: {e}")

    def _merge_delta_table(self, new_data_df: DataFrame, full_table_name: str, merge_key: str) -> None:
        """Perform upsert (merge) into an existing Delta table on the given key."""
        # Vue temporaire des updates
        temp_view = f"updates_{abs(hash(full_table_name)) % 10_000_000}"
        new_data_df.createOrReplaceTempView(temp_view)

        merge_sql = f"""
        MERGE INTO {full_table_name} AS target
        USING {temp_view} AS source
        ON target.{merge_key} = source.{merge_key}
        WHEN MATCHED THEN
        UPDATE SET *
        WHEN NOT MATCHED THEN
        INSERT *
        """
        self.spark.sql(merge_sql)

    def _get_table_row_count(self, full_table_name: str) -> int:
        """Return the number of rows in the target table if it exists."""
        try:
            count_df = self.spark.sql(f"SELECT COUNT(*) AS count FROM {full_table_name}")
            return count_df.collect()[0]["count"]
        except Exception:
            # Table doesn't exist or unreadable
            return 0


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
        synthetic_data["avg_price_per_room"] = pd.to_numeric(
            synthetic_data["avg_price_per_room"], errors="coerce"
        ).astype(np.float64)

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
            synthetic_data["arrival_year"] = np.random.randint(current_year - 2, current_year + 1, num_rows).astype(
                np.int32
            )

    return synthetic_data


@timeit
def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching Databricks schema with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
