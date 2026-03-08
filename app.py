from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

app = FastAPI(title="Semantic Search API")

with open("data/embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)
documents = data["documents"]

index = faiss.read_index("data/faiss_index.bin")
model = SentenceTransformer("all-MiniLM-L6-v2")

class StatsCache(SemanticCache):
    def __init__(self, threshold=0.85, max_size=100):
        super().__init__(threshold, max_size)
        self.hit_count = 0
        self.miss_count = 0

    def get(self, query_embedding):
        result = super().get(query_embedding)
        if result:
            self.hit_count += 1
        else:
            self.miss_count += 1
        return result

    def reset_stats(self):
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0

cache = StatsCache()

class QueryRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/query")
def query_endpoint(req: QueryRequest):
    query_vec = model.encode([req.query]).astype("float32")
    faiss.normalize_L2(query_vec)
    query_vec = query_vec[0]
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    cached_results = cache.get(query_vec_norm)
    if cached_results:
        matched_query = next(iter(cache.cache))  
        similarity = cache.cosine_similarity(query_vec_norm, cache.cache[matched_query][0])
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": str(matched_query[:50]),
            "similarity_score": float(similarity),
            "result": cached_results[:req.k],
            "dominant_cluster": None  
        }

    distances, indices = index.search(query_vec.reshape(1, -1), req.k)
    results = [documents[i] for i in indices[0]]
    cache.add(query_vec_norm, results)

    return {
        "query": req.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": None
    }

@app.get("/cache/stats")
def cache_stats():
    total_entries = len(cache.cache)
    hit_count = cache.hit_count
    miss_count = cache.miss_count
    hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0
    return {
        "total_entries": total_entries,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(hit_rate, 3)
    }

@app.delete("/cache")
def clear_cache():
    cache.reset_stats()
    return {"message": "Cache cleared and stats reset"}