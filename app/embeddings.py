import os
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDING_PATH = "data/embeddings.npy"


def generate_embeddings(texts):

    if os.path.exists(EMBEDDING_PATH):
        print("Loading saved embeddings...")
        return np.load(EMBEDDING_PATH)

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings)

    np.save(EMBEDDING_PATH, embeddings)

    print("Embeddings saved.")

    return embeddings


def embed_query(query):
    return model.encode([query])[0]