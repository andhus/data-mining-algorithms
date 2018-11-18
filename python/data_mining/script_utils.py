from __future__ import print_function, division

import argparse
import os

from time import time


DEFAULT_RESULTS_PATH = os.path.abspath(  # script outputs end up here.
    os.path.join(*([__file__] + [os.pardir] * 3 + ['results']))
)


def get_lsh_argument_parser(
    ndocs=100,
    nrows=10007,
    shingles_size=10,
    minhash_size=100,
    lsh_nbands=20,
):
    """Helper method for argument parsing for MinHash -> LSH script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ndocs',
        type=int,
        default=ndocs,
        help='the number of documents (first n) to analyse')
    parser.add_argument(
        '--nrows',
        type=int,
        default=nrows,
        help='range of output of hashing of shingles'
    )
    parser.add_argument(
        '-k', '--shingles-size',
        type=int,
        default=shingles_size,
        help='Size of the (character level) shingles to use'
    )
    parser.add_argument(
        '--minhash-size',
        type=int,
        default=minhash_size,
        help='size minhash signatures'
    )
    if lsh_nbands is not None:
        parser.add_argument(
            '--lsh-nbands',
            type=int,
            default=lsh_nbands,
            help='number of bands to use for LSH'
        )
    return parser


def get_frequent_itemsets_argument_parser(
    num_baskets_limit=None,
    max_set_size=None,
    min_support=0.005,
    min_confidence=0.10,
    print_limit=10
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-baskets-limit',
        type=int,
        default=num_baskets_limit,
        help='the number of (first n) baskets to analyse, leave empty for all')
    parser.add_argument(
        '--max-set-size',
        type=int,
        default=max_set_size,
        help='max size of frequent sets to identify, leave empty for all'
    )
    parser.add_argument(
        '--min-support',
        type=float,
        default=min_support,
        help='minimum support of sets and rules to identify'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=min_confidence,
        help='minimum confidence for rules to identify'
    )
    parser.add_argument(
        '--print-limit',
        type=float,
        default=print_limit,
        help='print maximum this number of items per printed data structure'
    )

    return parser


class _TimedContext(object):
    """Helper for timing code execution"""
    def __init__(self, time_f=time):
        self.time_f = time_f

    def __enter__(self):
        self.start = self.time_f()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = self.time_f()

    @property
    def elapsed(self):
        return self.end - self.start


def timer():
    """Creates a timing context.

    Usage:
        with timer() as something_timer:
            <run somethings>
        print(something_timer.elapsed)
    """
    return _TimedContext()


def mkdirp(path):
    """Recursively creates directories to the specified path"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('{} exists and is not a directory'.format(path))
    else:
        os.makedirs(path)
