
# Semantic Search Cache System

Lightweight semantic search system built on the **20 Newsgroups dataset** that combines:

- Vector embeddings for semantic understanding
- Fuzzy clustering to uncover latent topic structure
- A custom **semantic cache** built from first principles
- A **FastAPI service** exposing the system as a live API

The system avoids redundant computation by recognizing **semantically similar queries**, not just identical strings.

---

# Repository Structure

```

semantic-search-cache-system
│
├── app
│   ├── cache.py
│   ├── clustering.py
│   ├── embeddings.py
│   ├── main.py
│   └── search.py
│
├── data
│   ├── 20_newsgroups
│   ├── embeddings.npy
│   ├── faiss.index
│   └── preprocess.py
│
├── logs
│   └── query_log.json
│
├── notebooks
│   └── cluster_analysis.py
│
├── Dockerfile
├── requirements.txt
├── test_pipeline.py
└── README.md

```

---

# Dataset

Dataset used: **20 Newsgroups**

Source:  
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

The dataset contains roughly **20,000 news posts** across **20 categories** including politics, science, religion, sports, etc.

The raw dataset contains noise such as:

- inconsistent whitespace
- formatting artifacts
- metadata fragments

During preprocessing:

- documents are loaded from category folders
- whitespace normalization is applied
- unreadable files are skipped

Cleaning is intentionally minimal because semantic models often perform better when contextual structure is preserved.

---

# Part 1 — Embedding & Vector Database

## Embedding Model

Model used:

```

sentence-transformers/all-MiniLM-L6-v2

```

Reasons:

- strong semantic performance
- lightweight (384 dimensional embeddings)
- fast inference suitable for real-time API systems
- widely used in production search systems

Embeddings are generated once and stored locally:

```

data/embeddings.npy

```

On subsequent runs the system **loads the saved embeddings instead of recomputing them**.

---

## Vector Database

Vector search is implemented using **FAISS**.

Index type:

```

IndexFlatL2

```

Reasons:

- simple and exact search
- efficient for medium datasets (~20k vectors)
- no training phase required
- ideal for prototyping semantic retrieval systems

The FAISS index is persisted:

```

data/faiss.index

```

If the index exists it is loaded automatically.

Search returns the **top-k most similar documents**.

---

# Part 2 — Fuzzy Clustering

The dataset contains **overlapping semantic topics**.

Example:

A post about gun legislation relates to:

- politics
- firearms
- legal discussion

A hard cluster assignment would lose this nuance.

Therefore **Gaussian Mixture Models (GMM)** are used to produce **probabilistic cluster membership**.

```

P(cluster | document)

```

Each document receives a distribution such as:

```

cluster_0 : 0.52
cluster_1 : 0.31
cluster_2 : 0.12
cluster_3 : 0.05

```

---

## Number of Clusters

The dataset has 20 labelled groups, but real semantic structure is often **lower dimensional**.

The model uses:

```

n_clusters = 10

```

Reasoning:

- captures broader semantic themes
- prevents fragmentation of clusters
- produces meaningful probability distributions
- improves cache locality (important for Part 3)

Cluster analysis is performed using:

```

notebooks/cluster_analysis.py

```

The analysis script:

- prints representative documents per cluster
- identifies **boundary documents**
- shows where the model is uncertain

Boundary documents are identified by comparing the difference between the top two cluster probabilities.

Small differences indicate **semantic overlap**, which is common in real text corpora.

---

# Part 3 — Semantic Cache

Traditional caches rely on **exact key matches**.

Example:

```

"What causes black holes?"
"How are black holes formed?"

```

Both queries mean the same thing but would miss in a normal cache.

This system instead caches based on **semantic similarity of query embeddings**.

---

## Cache Design

The cache stores entries in a structure grouped by cluster:

```

cluster_id → [cache entries]

```

Each entry contains:

```

{
query,
embedding,
result,
cluster
}

```

Cluster partitioning improves lookup efficiency by reducing the number of comparisons.

Instead of comparing against **all cached queries**, the system only compares within the **dominant cluster**.

---

## Cache Lookup

Steps:

1. Embed incoming query
2. Determine its dominant cluster
3. Compare against cached queries within that cluster
4. Compute cosine similarity

If similarity exceeds the threshold:

```

similarity >= threshold

```

the cache returns the stored result.

Otherwise a fresh search is executed and stored.

---

## Tunable Parameter

The key parameter controlling cache behavior:

```

similarity threshold = 0.88

```

This determines how strict the cache is.

Lower threshold:

- higher cache hit rate
- greater risk of returning slightly mismatched results

Higher threshold:

- more precise matches
- fewer cache hits

The system behavior changes noticeably across thresholds:

| Threshold | Behavior |
|----------|----------|
| 0.75 | aggressive caching |
| 0.85 | balanced |
| 0.90+ | conservative |

The chosen value **0.88** balances precision with reuse.

---

## Cache Metrics

The cache tracks:

```

total_entries
hit_count
miss_count
hit_rate

````

These statistics are accessible via the API.

---

# Part 4 — FastAPI Service

The system exposes three endpoints.

---

# System Workflow

The full request lifecycle in the system:

```

User Query
↓
Query Embedding (Sentence Transformer)
↓
Cluster Prediction (Gaussian Mixture Model)
↓
Semantic Cache Lookup (within dominant cluster)

**If cache hit**

Return cached result

**If cache miss**

↓
FAISS Vector Search
↓
Retrieve top-k documents
↓
Store result in semantic cache
↓
Return results to user

```

This workflow ensures that repeated or semantically similar queries avoid recomputation while still retrieving relevant documents when new queries appear.

---



---

## POST /query

Accepts:

```json
{
  "query": "natural language query"
}
````

Response:

```json
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "results": [...],
  "dominant_cluster": 3,
  "cluster_distribution": {...}
}
```

Workflow:

1. Embed query
2. Predict cluster distribution
3. Attempt semantic cache lookup
4. On miss → perform FAISS search
5. Store result in cache
6. Log query

---

## GET /cache/stats

Returns cache statistics:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

## DELETE /cache

Clears cache and resets metrics.

---

# Query Logging

All queries are recorded in:

```
logs/query_log.json
```

Each entry includes:

```
{
  query,
  time,
  cache_hit
}
```

This allows analysis of:

* cache effectiveness
* repeated query patterns
* semantic query reuse

---

# Running the Project

## 1. Clone the repository

```
git clone <repo-link>
cd semantic-search-cache-system
```

---

## 2. Create virtual environment

```
python -m venv venv
```

Activate:

Linux / Mac

```
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

---

## 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 4. Start the FastAPI server

```
uvicorn app.main:app --reload
```

Server runs at:

```
http://localhost:8000
```

Interactive API docs:

```
http://localhost:8000/docs
```

---

# Pipeline Test

To verify dataset and embeddings pipeline:

```
python test_pipeline.py
```

This checks:

* dataset loading
* embedding generation
* embedding dimensionality

---

# Cluster Analysis

To inspect cluster behavior:

```
python notebooks/cluster_analysis.py
```

The script shows:

* representative documents per cluster
* boundary documents with ambiguous cluster membership
* probability distributions

This validates that clustering captures **meaningful semantic structure**.

---

# Docker

Build container:

```
docker build -t semantic-search .
```

Run container:

```
docker run -p 8000:8000 semantic-search
```

The container starts the FastAPI server automatically.

---

# Design Summary

Key design principles used in this system:

**Semantic understanding**

* transformer embeddings

**Efficient retrieval**

* FAISS vector search

**Probabilistic topic structure**

* Gaussian mixture clustering

**Query reuse**

* semantic similarity cache

**Scalable lookup**

* cluster-partitioned cache

This combination enables the system to behave closer to **how humans interpret similar questions**, rather than relying on exact string matching.

```
