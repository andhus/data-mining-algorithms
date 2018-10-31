from __future__ import print_function, division

from data_mining.data import get_news_groups_documents
from data_mining.compare import (
    get_shingles,
    get_hash_set,
    MinHashing,
    get_jaccard_similarity,
    get_estimated_jaccard_similarity_from_signatures
)

if __name__ == '__main__':
    N_ROWS = 10973
    K = 10

    def primary_hash(s):
        """Pythons default hash(str) returns value in range
            [-(sys.maxint + 1):sys.maxint]
        """
        return hash(s) % N_ROWS

    documents = list(get_news_groups_documents(mini=True, first_n=100))
    shingle_sequences = [get_shingles(doc, k=K) for doc in documents]
    hash_sets = [get_hash_set(ss, hash_f=primary_hash) for ss in shingle_sequences]
    min_hashing = MinHashing(n_rows=N_ROWS, n_hash_fs=100)
    signatures = min_hashing(hash_sets)
