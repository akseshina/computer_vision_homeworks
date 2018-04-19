# based on https://github.com/kayzhu/LSHash

import numpy as np
from bitarray import bitarray

class LSHash(object):
    """
    :param hash_size:
        The length of the resulting binary hash in integer
    :param input_dim:
        The dimension of the input vector
    """
    def __init__(self, hash_size, input_dim):
        np.random.seed(42)
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.uniform_planes = np.random.randn(self.hash_size, self.input_dim)
        self.hash_table = dict()

    def _hash(self, planes, input_point):
        projections = np.dot(planes, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def index(self, input_point, extra_data):
        key = self._hash(self.uniform_planes, input_point)
        value = (tuple(input_point), extra_data)
        self.hash_table.setdefault(key, []).append(value)

    def query(self, query_point, num_results=10):

        binary_hash = self._hash(self.uniform_planes, query_point)
        candidates = set()
        max_dist = 3

        for key in self.hash_table.keys():
            distance = LSHash.hamming_dist(key, binary_hash)
            if distance <= max_dist:
                candidates.update(self.hash_table.get(key, []))

        while len(candidates) < 10:
            max_dist += 1
            for key in self.hash_table.keys():
                distance = LSHash.hamming_dist(key, binary_hash)
                if distance == max_dist:
                    candidates.update(self.hash_table.get(key, []))

        candidates = [(ix, self.chi2_distance(query_point, ix[0]))
                      for ix in candidates]
        candidates.sort(key=lambda x: x[1])

        candidates_items = list(zip(*candidates))[0]
        candidates_items = list(zip(*candidates_items))[1]

        return candidates_items[:num_results]

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def chi2_distance(histA, histB, eps=1e-10):
        return np.sum((histA - histB) ** 2 / (histA + histB + eps))
