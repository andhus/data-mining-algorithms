from __future__ import print_function, division

import os
from glob import glob

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3 + ['datasets']))
)


def get_news_groups_documents(mini=False, first_n=None):
    main_dir = os.path.join(
        DEFAULT_DATA_PATH,
        'NewsGroups',
        '20_newsgroups' if not mini else 'mini_newsgroups'
    )
    doc_paths = glob(main_dir + '/*/*')
    for i, doc_path in enumerate(doc_paths):
        if first_n is not None and i >= first_n:
            break
        with open(doc_path) as f:
            doc = f.read()
        yield doc

