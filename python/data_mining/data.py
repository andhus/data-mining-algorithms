from __future__ import print_function, division

import os

from glob import glob
import subprocess

PROJECT_ROOT = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3))
)
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets')
BASH_SCRIPTS_PATH = os.path.join(PROJECT_ROOT, 'scripts')


class LinesIterator(object):
    """Iterates over and optionally post process the lines of a file.

    Args:
        filepath (str): file to iterate over.
        limit (int | None): If provided, only iterate over first `limit` lines.
        postprocess (f: str -> object | None): A function to apply to each line
            before output.
    """
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


def _require_dataset(name, filepath, script, auto_download=True):
    if not os.path.exists(filepath):
        if auto_download:
            process = subprocess.Popen(
                os.path.join(BASH_SCRIPTS_PATH, script),
                stdout=subprocess.PIPE
            )
            output, error = process.communicate()
            if error or not os.path.exists(filepath):
                raise IOError(
                    'Failed to auto-downloading {}, no such file or directly: {}. '
                    'Try to run: ./data-mining-algorithms/scripts/{}` '
                    'manually to download the data.'.format(name, filepath, script)
                )
        else:
            raise IOError(
                'Failed to access {}, no such file or directly: {}. '
                'Try to run: ./data-mining-algorithms/scripts/{}` '
                'to download the data.'.format(name, filepath, script)
            )


def get_news_groups_documents(mini=False, first_n=None, auto_download=True):
    """Load NewsGroups data.

    Args:
        mini (bool): If True, use the "mini" version of the dataset.
        first_n (int|None): If provided, limit the number of documents to this
            number.
        auto_download (bool): If True and data can be found it is automatically
            downloaded.

    Returns:
        (Iterator(str)): One string per document.
    """
    maindir = os.path.join(
        DEFAULT_DATA_PATH,
        'NewsGroups',
        '20_newsgroups' if not mini else 'mini_newsgroups'
    )
    _require_dataset(
        name='NewsGroups',
        filepath=maindir,
        script='download_news_groups.sh',
        auto_download=auto_download
    )
    doc_paths = glob(maindir + '/*/*')

    # put generation in separate function to make code above execute before
    # trying to get first item.
    def generator():
        for i, doc_path in enumerate(doc_paths):
            if first_n is not None and i >= first_n:
                break
            with open(doc_path) as f:
                doc = f.read()
            yield doc

    return generator()


def get_frequent_itemset(first_n=None, auto_download=True):
    """Load Frequent Itemset Mining data.

    Args:
        first_n (int | None): If provided, limit the number of baskets to this
            number.
        auto_download (bool): If True and data can be found it is automatically
            downloaded.

    Returns:
        (Iterator(tuple(int))): The basket sets.

    """
    subset = 'T10I4D100K'
    filepath = os.path.join(
        DEFAULT_DATA_PATH,
        'FrequentItemsetMining',
        subset + ".dat"
    )
    _require_dataset(
        name='FrequentItemsetMining',
        filepath=filepath,
        script='download_frequent_itemset.sh',
        auto_download=auto_download
    )
    return LinesIterator(
        filepath,
        limit=first_n,
        postprocess=lambda line: tuple([int(v) for v in line.split()])
    )


def get_facebook_links(first_n=None, auto_download=True, unique=True):
    """

    Args:
        first_n (int | None): If provided, limit the number of baskets to this
            number.
        auto_download (bool): If True and data can be found it is automatically
            downloaded.

    Returns:
        (Iterator(tuple(int))): The basket sets.
    """
    org_filepath = os.path.join(
        DEFAULT_DATA_PATH,
        'FacebookLinks',
        'facebook-links.txt'
    )
    _require_dataset(
        name='FacebookLinks',
        filepath=org_filepath,
        script='download_facebook_links.sh',
        auto_download=auto_download
    )
    unique_filepath = os.path.join(
        DEFAULT_DATA_PATH,
        'FacebookLinks',
        'facebook-links-unique.txt'
    )
    post_process_f = lambda line: tuple(
        sorted([int(v) for v in line.split()[:2]])
    )
    if unique and not os.path.exists(unique_filepath):
        seen =set([])
        with open(org_filepath) as f_org:
            with open(unique_filepath, 'w') as f_unique:
                for line in f_org:
                    res = post_process_f(line)
                    if res not in seen:
                        seen.add(res)
                        f_unique.write(line)

    return LinesIterator(
        org_filepath if not unique else unique_filepath,
        limit=first_n,
        postprocess=post_process_f
    )
