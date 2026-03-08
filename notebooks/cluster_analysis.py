import sys
import os

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from data.preprocess import load_dataset
from app.embeddings import generate_embeddings
from app.clustering import FuzzyCluster

def show_cluster_examples(documents, probs, top_n=3):

    num_clusters = probs.shape[1]

    for c in range(num_clusters):

        print("\n==============================")
        print(f"Cluster {c}")
        print("==============================")

        top_docs = probs[:, c].argsort()[-top_n:][::-1]

        for i in top_docs:
            print("\nProbability:", probs[i][c])
            print(documents[i][:300])


def show_boundary_documents(documents, probs, top_n=5):

    uncertainty = np.abs(
        np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    )

    boundary_docs = np.argsort(uncertainty)[:top_n]

    print("\n===== Boundary Documents =====")

    for i in boundary_docs:

        print("\nDocument:")
        print(documents[i][:300])

        print("\nCluster distribution:")

        top_clusters = np.argsort(probs[i])[-3:][::-1]

        for c in top_clusters:
            print(f"cluster_{c} :", probs[i][c])


def main():

    print("Loading dataset...")
    documents, labels = load_dataset()

    print("Generating embeddings...")
    embeddings = generate_embeddings(documents)

    print("Training fuzzy clustering model...")
    cluster_model = FuzzyCluster(n_clusters=10)
    cluster_model.fit(embeddings)

    print("Getting cluster probabilities...")
    cluster_probs = cluster_model.model.predict_proba(embeddings)

    print("\nEmbedding shape:", embeddings.shape)
    print("Cluster probability shape:", cluster_probs.shape)

    show_cluster_examples(documents, cluster_probs)

    show_boundary_documents(documents, cluster_probs)


if __name__ == "__main__":
    main()