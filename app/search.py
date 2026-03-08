import os
import numpy as np
import faiss


class SemanticSearch:

    def __init__(self, embeddings, documents):

        self.documents = documents
        self.index_path = "data/faiss.index"

        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]

        if os.path.exists(self.index_path):

            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)

        else:

            print("Building new FAISS index...")

            self.index = faiss.IndexFlatL2(dim)

            self.index.add(embeddings)

            faiss.write_index(self.index, self.index_path)

            print("FAISS index saved.")

    def search(self, query_embedding, k=5):

        query_embedding = np.array(query_embedding)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for i, idx in enumerate(indices[0]):
            results.append({
                "doc_id": int(idx),
                "score": float(1 - distances[0][i]),
                "text": self.documents[idx][:200]
            })

        return results