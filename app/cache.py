import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.88):

        self.cache = {}
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding, cluster):

        if cluster not in self.cache:
            self.miss_count += 1
            return False, None, 0

        query_embedding = query_embedding.reshape(1, -1)

        best_score = 0
        best_entry = None

        for entry in self.cache[cluster]:

            stored = np.array(entry["embedding"]).reshape(1, -1)

            score = cosine_similarity(query_embedding, stored)[0][0]

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            self.hit_count += 1
            return True, best_entry, best_score

        self.miss_count += 1
        return False, None, best_score

    def add(self, query, embedding, result, cluster):

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append({
            "query": query,
            "embedding": embedding.tolist(),
            "result": result,
            "cluster": cluster
        })

    def stats(self):

        total_entries = sum(len(v) for v in self.cache.values())

        hit_rate = 0

        if (self.hit_count + self.miss_count) > 0:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count)

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0