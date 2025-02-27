"""
spark_generate_embeddings.py
Generates embeddings using a Transformer model via Spark Pandas UDF.
"""

import argparse
import torch
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model

@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def embed_text_batch(texts: pd.Series) -> pd.Series:
    tok, mdl = load_model()

    texts_list = texts.fillna("").tolist()
    encodings = tok(
        texts_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = mdl(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return pd.Series(embeddings.cpu().numpy().tolist())

def main(input_path, output_path):
    spark = (SparkSession.builder
             .appName("SparkGenerateEmbeddings")
             .getOrCreate())

    df = spark.read.parquet(input_path)
    # Assume df: [doc_id, cleaned_text, ...]

    df_embedded = df.withColumn("embedding", embed_text_batch(F.col("cleaned_text")))
    df_embedded.write.mode("overwrite").parquet(output_path)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cleaned data input path")
    parser.add_argument("--output", required=True, help="Output path for embeddings parquet")
    args = parser.parse_args()

    main(args.input, args.output)

