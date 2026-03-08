from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyCluster:

    def __init__(self, n_clusters=10):
        self.model = GaussianMixture(n_components=n_clusters)

    def fit(self, embeddings):
        self.model.fit(embeddings)

    def predict_cluster(self, embedding):
        import numpy as np

        embedding = np.array(embedding)

        # Convert 1D embedding → 2D
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        probs = self.model.predict_proba(embedding)[0]

        dominant_cluster = int(np.argmax(probs))

        cluster_distribution = {
            f"cluster_{i}": float(p) for i, p in enumerate(probs)
        }

        return dominant_cluster, cluster_distribution