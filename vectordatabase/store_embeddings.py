"""
store_embeddings.py
Load embeddings from Parquet, insert them into a Milvus collection.
"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import glob
import os

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Document Embeddings")
    collection = Collection(name=collection_name, schema=schema)
    return collection

def main(embedding_dir, milvus_host, milvus_port, collection_name):
    # Connect to Milvus
    connections.connect("default", host=milvus_host, port=milvus_port)

    # Assuming we know embedding dimension from prior knowledge
    dim = 384  # for "all-MiniLM-L6-v2"
    collection = create_collection(collection_name, dim)

    # Optionally create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    parquet_files = glob.glob(os.path.join(embedding_dir, "*.parquet"))
    for file in parquet_files:
        table = pq.read_table(file)
        df = table.to_pandas()
        # Expect columns: doc_id, embedding
        doc_ids = df["doc_id"].astype(str).tolist()
        embeddings = df["embedding"].tolist()

        # Insert into Milvus
        data_to_insert = [doc_ids, embeddings]
        collection.insert(data_to_insert)

    # Flush to persist
    collection.load()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--milvus_host", default="localhost")
    parser.add_argument("--milvus_port", default="19530")
    parser.add_argument("--collection_name", default="documents")
    args = parser.parse_args()

    main(
        args.embedding_dir,
        args.milvus_host,
        args.milvus_port,
        args.collection_name
    )

# You can choose Faiss (local or distributed), Milvus, Weaviate, Pinecone, or OpenSearch for vector storage. Below is an example for Milvus (a popular open-source vector DB), but adapt as needed.
