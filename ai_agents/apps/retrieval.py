"""
retrieval.py
Retrieve top-k similar documents from Milvus for a user query.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection
from ai_agent.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class MilvusRetriever:
    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.collection = Collection(self.collection_name)
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(settings.EMBEDDING_MODEL).cuda() if torch.cuda.is_available() else AutoModel.from_pretrained(settings.EMBEDDING_MODEL)
        self.model.eval()

    def embed_query(self, query: str):
        enc = self.tokenizer([query], return_tensors="pt", padding=True, truncation=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            output = self.model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            embedding = output.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]
        return embedding

    def search(self, query: str, top_k=3):
        embedding = self.embed_query(query)
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )
        logger.info(f"Search completed for query='{query}', top_k={top_k}")
        return results[0]  # single-vector search

