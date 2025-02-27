"""
spark_cleaning.py
Spark Job: Clean and normalize ingested data, remove duplicates, handle missing columns, etc.
"""

import argparse
from pyspark.sql import SparkSession, functions as F

def main(input_path, output_path):
    spark = (SparkSession.builder
             .appName("SparkDataCleaning")
             .getOrCreate())

    df = spark.read.parquet(input_path)

    # Example cleaning transformations
    df_cleaned = (
        df
        .dropDuplicates(["text"])
        .filter(F.col("text").isNotNull())
        # Additional cleaning logic
    )

    df_cleaned.write.mode("overwrite").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)
