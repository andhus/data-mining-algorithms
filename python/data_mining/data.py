from __future__ import print_function, division

import os
from glob import glob

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3 + ['datasets']))
)


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
