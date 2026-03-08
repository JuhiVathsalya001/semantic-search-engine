import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_cache import SemanticCache  
cache = SemanticCache(threshold=0.85, max_size=100)

with open("data/embeddings_data.pkl", "rb") as f:
    data = pickle.load(f)

documents = data["documents"]

index = faiss.read_index("data/faiss_index.bin")
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, k=5):
    query_vec = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)  
    query_vec = query_vec[0]

    cached_results = cache.get(query_vec)
    if cached_results:
        print("Cache hit!")
        return cached_results

    print("Cache miss → running FAISS search")
    distances, indices = index.search(query_vec.reshape(1, -1), k)
    results = [documents[i] for i in indices[0]]

    cache.add(query_vec, results)

    return results


if __name__ == "__main__":
    query = input("Enter search query: ")
    results = search(query)

    print("\nTop Results:\n")
    seen = set()
    rank = 1
    for r in results:
        if r not in seen:
            print(f"Result {rank}:")
            print(r[:300])
            print("-----")
            seen.add(r)
            rank += 1