from __future__ import print_function, division

from pprint import pprint
from data_mining.data import get_news_groups_documents
from data_mining.compare import (
    get_shingles,
    get_hash_set,
    MinHashing,
    LocalitySensitiveHashing,
    get_jsim_and_approx_jsim
)
from data_mining.script_utils import get_lsh_argument_parser


if __name__ == '__main__':
    parser = get_lsh_argument_parser(
        ndocs=100,
        nrows=10007,
        shingles_size=10,
        minhash_size=100,
        lsh_nbands=20,
    )
    args = parser.parse_args()
    print("Running overview script with args:")
    print(args)

    def primary_hash(s):
        """Pythons default hash(str) returns value in range
            [-(sys.maxint + 1):sys.maxint]
        """
        return hash(s) % args.nrows

    print("loading documents")
    documents = list(get_news_groups_documents(first_n=args.ndocs))

    print("extracting shingles")
    shingle_sequences = [
        get_shingles(doc, k=args.shingles_size) for doc in documents
    ]
    hash_sets = [get_hash_set(ss, hash_f=primary_hash) for ss in shingle_sequences]

    print("computing minhash signatures")
    mh = MinHashing(n_rows=args.nrows, n_hash_fs=args.minhash_size)
    signatures = mh(hash_sets)

    print("finding similar documents using LSH")
    lsh = LocalitySensitiveHashing(n_bands=args.lsh_nbands)
    top_similar_index = lsh.get_top_similar_index(signatures)
    top_similarities_index = get_jsim_and_approx_jsim(
        top_similar_index,
        hash_sets,
        signatures
    )
    print("most similar documents and their Jaccard and approx. Jaccard similarity")
    pprint(top_similarities_index)
