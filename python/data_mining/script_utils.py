from __future__ import print_function, division

import argparse
import os

from time import time


DEFAULT_RESULTS_PATH = os.path.abspath(
    os.path.join(*([__file__] + [os.pardir] * 3 + ['results']))
)


def get_lsh_argument_parser(
    ndocs=100,
    nrows=10007,
    shingles_size=10,
    minhash_size=100,
    lsh_nbands=20,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ndocs",
        type=int,
        default=ndocs,
        help="the number of documents (first n) to analyse")
    parser.add_argument(
        "--nrows",
        type=int,
        default=nrows,
        help="range of output of hashing of shingles"
    )
    parser.add_argument(
        "-k", "--shingles-size",
        type=int,
        default=shingles_size,
        help="TODO"
    )
    parser.add_argument(
        "--minhash-size",
        type=int,
        default=minhash_size,
        help="TODO"
    )
    parser.add_argument(
        "--lsh-nbands",
        type=int,
        default=lsh_nbands,
        help="TODO"
    )
    return parser


class _TimedContext(object):

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
    return _TimedContext()


def mkdirp(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError("{} exists and is not a directory".format(path))
    else:
        os.makedirs(path)
