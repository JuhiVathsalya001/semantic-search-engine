## Overview:
This project implements a ***semantic search engine*** over a document corpus using sentence embeddings, FAISS for similarity search, and a semantic cache with LRU eviction. The API is served with FastAPI, providing endpoints to query the search engine and manage the cache.

## Key features:
1.Query embedding using SentenceTransformer (all-MiniLM-L6-v2)  
2.Top-k similar document retrieval with FAISS  
3.Semantic cache with cosine similarity threshold and LRU eviction  
4.Cache statistics and clearing functionality  

## Folder Structure
.
├── data/
│   ├── embeddings_data.pkl   # Precomputed document embeddings & documents
    └── faiss_index.bin # FAISS index of embeddings
├── semantic_cache.py       # Semantic cache implementation
├── app.py                # FastAPI application
├── search.py     # Search logic
├── requirements.txt         # Python dependencies
└── README.md

## Setup
1. Create a virtual environment  
python -m venv venv  
source venv/bin/activate      # Linux/macOS  
venv\Scripts\activate         # Windows  

2. Install dependencies
pip install -r requirements.txt

3. Start the FastAPI server
uvicorn app:app --reload

Server will run at: http://127.0.0.1:8000
