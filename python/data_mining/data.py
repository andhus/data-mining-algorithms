from __future__ import print_function, division

import os
from glob import glob

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3 + ['datasets']))
)


class LinesIterator(object):

    def __init__(self, filepath, limit=None, postprocess=None):
        self.filepath = filepath
        self.limit = limit
        self.postprocess = postprocess or (lambda x: x)
        self._len = limit

    def __iter__(self):
        with open(self.filepath) as f:
            for i, line in enumerate(f):
                if self.limit and i == self.limit:
                    break
                yield self.postprocess(line)
            self._len = i + 1

    def __len__(self):
        if self._len is None:
            with open(self.filepath) as f:
                for i, _ in enumerate(f):
                    pass
            self._len = i + 1

        return self._len


def get_news_groups_documents(mini=False, first_n=None):
    """Load NewsGroups data.

    Args:
        mini (bool): If True, use the "mini" version of the dataset.
        first_n: If provided, limit the number of documents to this number.

    Returns:
        (Iterator(str)): One string per document.
    """
    main_dir = os.path.join(
        DEFAULT_DATA_PATH,
        'NewsGroups',
        '20_newsgroups' if not mini else 'mini_newsgroups'
    )
    if not os.path.exists(main_dir):
        raise IOError(
            "Could not find NewsGroups dataset. Run: "
            "`./data-mining-algorithms/scripts/download_news_groups.sh` "
            "to download the data."
        )
    doc_paths = glob(main_dir + '/*/*')
    for i, doc_path in enumerate(doc_paths):
        if first_n is not None and i >= first_n:
            break
        with open(doc_path) as f:
            doc = f.read()
        yield doc


def get_frequent_itemset(first_n=None):
    subset = 'T10I4D100K'
    filepath = os.path.join(
        DEFAULT_DATA_PATH,
        'FrequentItemsetMining',
        subset + ".dat"
    )
    if not os.path.exists(filepath):
        raise IOError(
            "Could not find Frequent Itemset Mining dataset. Run: "
            "`./data-mining-algorithms/scripts/download_frequent_itemset.sh` "
            "to download the data."
        )
    return LinesIterator(
        filepath,
        limit=first_n,
        postprocess=lambda line: tuple([int(v) for v in line.split()])
    )
