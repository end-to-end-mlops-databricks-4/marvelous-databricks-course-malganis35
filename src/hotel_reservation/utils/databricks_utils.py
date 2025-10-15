"""Databricks utilities module.

Provides helper functions to create a Databricks-connected Spark session
that works with both Free (Serverless) and Premium (Cluster) workspaces.

Usage:
    from hotel_reservation.utils.databricks_utils import create_spark_session

    spark = create_spark_session()
"""

import json
import os
import subprocess
import time

import pyspark
from loguru import logger
from pyspark.sql import SparkSession

# Try importing DatabricksSession safely (for environments without Connect)
try:
    from databricks.connect import DatabricksSession
    DATABRICKS_AVAILABLE = True
except ImportError:
    DatabricksSession = None
    DATABRICKS_AVAILABLE = False

def create_spark_session() -> "pyspark.sql.SparkSession":
    """Create a Spark session connected to Databricks via Databricks Connect.

    Automatically determines whether to use Serverless or Cluster compute
    based on environment variables loaded from .env.

    Expected environment variables:
        - DATABRICKS_COMPUTE: "serverless" or "cluster"
        - DATABRICKS_CLUSTER_ID: (required if using cluster)
        - DATABRICKS_HOST, DATABRICKS_TOKEN: set via .env or system env

    Returns:
        pyspark.sql.SparkSession

    """

    compute_mode = os.getenv("DATABRICKS_COMPUTE", "serverless").lower()
    cluster_id = os.getenv("DATABRICKS_CLUSTER_ID")

    # Attempting to initialize local Spark session
    try:
        logger.info("Attempting to initialize local Spark session")
        spark = SparkSession.builder.getOrCreate()
        logger.info("✅ Local Spark session initialized successfully")
        return spark
    except Exception as e:
        logger.warning(f"⚠️ Falling back to Databricks Connect due to: {e}")

    # Connection Databricks Connect
    if not DATABRICKS_AVAILABLE:
        raise ImportError(
            "Databricks Connect is not installed. Run: pip install databricks-connect"
        )

    builder = DatabricksSession.builder
    if compute_mode == "cluster" and cluster_id:
        logger.info(f"🔗 Connecting to Databricks cluster: {cluster_id}")
        spark = builder.remote(cluster_id=cluster_id).getOrCreate()
    elif compute_mode == "serverless":
        logger.info("🔗 Connecting to Databricks Serverless compute")
        spark = builder.remote(serverless=True).getOrCreate()
    else:
        logger.warning("⚠️ No compute specified — defaulting to serverless")
        spark = builder.remote(serverless=True).getOrCreate()

    # Force an active session
    active = SparkSession.getActiveSession()
    if not active:
        logger.warning("⚠️ No active SparkSession detected — forcing creation…")
        spark = builder.getOrCreate()
    else:
        spark = active

    # Check that the cluster is responding
    for i in range(6):
        try:
            logger.info("⏳ Verifying Spark connection with a tiny query…")
            spark.range(1, 1).collect()
            logger.info("✅ Spark session confirmed active and cluster responsive")
            break
        except Exception as e:
            logger.warning(f"Cluster not ready yet ({type(e).__name__}: {e}); retrying in 10 s…")
            time.sleep(10)
    else:
        raise RuntimeError("Databricks cluster not ready after several attempts.")

    logger.info("✅ Spark session initialized successfully via Databricks Connect")
    return spark


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


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment.

    :return: True if running in Databricks, False otherwise.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ
