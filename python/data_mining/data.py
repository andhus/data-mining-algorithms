from __future__ import print_function, division

import os
from glob import glob

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3 + ['datasets']))
)


def get_news_groups_documents(mini=False):
    main_dir = os.path.join(
        DEFAULT_DATA_PATH,
        'NewsGroups',
        '20_newsgroups' if not mini else 'mini_newsgroups'
    )
    doc_paths = glob(main_dir + '/*/*')
    for doc_path in doc_paths:
        with open(doc_path) as f:
            doc = f.read()
        yield doc
