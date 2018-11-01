from __future__ import print_function, division


import argparse


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
