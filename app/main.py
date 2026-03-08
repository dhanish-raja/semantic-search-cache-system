from fastapi import FastAPI
from pydantic import BaseModel

from data.preprocess import load_dataset
from app.embeddings import generate_embeddings, embed_query
from app.search import SemanticSearch
from app.clustering import FuzzyCluster
from app.cache import SemanticCache

import json
import os
from datetime import datetime

app = FastAPI()

documents = None
labels = None
embeddings = None
vector_db = None
cluster_model = None
cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():

    global documents, labels, embeddings, vector_db, cluster_model

    print("Loading dataset...")
    documents, labels = load_dataset()

    print("Generating embeddings...")
    embeddings = generate_embeddings(documents)

    print("Building vector index...")
    vector_db = SemanticSearch(embeddings, documents)

    print("Training clustering model...")
    cluster_model = FuzzyCluster(n_clusters=10)
    cluster_model.fit(embeddings)

    # ensure logs folder exists
    os.makedirs("logs", exist_ok=True)

    print("System ready.")


@app.post("/query")
def query_api(req: QueryRequest):

    query = req.query

    query_embedding = embed_query(query)

    cluster, cluster_distribution = cluster_model.predict_cluster(query_embedding)

    hit, entry, score = cache.lookup(query_embedding, cluster)

    if hit:

        cache_hit = True
        results = entry["result"]
        matched_query = entry["query"]

    else:

        cache_hit = False

        results = vector_db.search(query_embedding, k=5)

        matched_query = None

        cache.add(query, query_embedding, results, cluster)

    log_entry = {
        "query": query,
        "time": str(datetime.now()),
        "cache_hit": cache_hit
    }

    with open("logs/query_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "query": query,
        "cache_hit": cache_hit,
        "matched_query": matched_query,
        "similarity_score": float(score) if hit else 0,
        "results": results,
        "dominant_cluster": int(cluster),
        "cluster_distribution": cluster_distribution
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}