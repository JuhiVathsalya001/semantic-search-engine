import numpy as np
from collections import OrderedDict

class SemanticCache:

    def __init__(self, threshold=0.85, max_size=100):
        self.cache = OrderedDict()
        self.threshold = threshold
        self.max_size = max_size


    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def get(self, query_embedding):
        for key in list(self.cache.keys()):
            cached_embedding, results = self.cache[key]
            similarity = self.cosine_similarity(query_embedding, cached_embedding)

            if similarity >= self.threshold:
                self.cache.move_to_end(key)

                return results

        return None


    def add(self, query_embedding, results):

        key = tuple(query_embedding)

        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = (query_embedding, results)

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)