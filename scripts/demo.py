"""Demo script to showcase hotel_resa package functionality."""

# %% Databricks notebook source
import pretty_errors  # noqa: F401
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

df = spark.read.table("samples.nyctaxi.trips")
df.show(5)

# COMMAND ----------

print(list(df.columns))

# COMMAND ----------
