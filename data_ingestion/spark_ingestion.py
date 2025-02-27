"""
spark_ingestion.py
Spark Job: Ingest raw data (files or streams) and write them to a data lake in Parquet.
"""

import argparse
from pyspark.sql import SparkSession

def main(input_path, output_path):
    spark = (SparkSession.builder
             .appName("SparkDataIngestion")
             .getOrCreate())

    # Example: read a raw CSV or JSON
    df = spark.read.option("header", True).csv(input_path)
    # Or a streaming read with .readStream if you have streaming data.

    # Potential transformations or schema standardization here.

    # Write to Parquet for efficient downstream processing
    df.write.mode("append").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path or endpoint for raw data")
    parser.add_argument("--output", required=True, help="Output path for ingested data")
    args = parser.parse_args()

    main(args.input, args.output)
    main(args.input, args.output)
