from __future__ import print_function, division

from nose.tools import assert_equal

primes_up_to_1009 = [
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
    251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
    349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
    443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643,
    647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751,
    757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
    863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977,
    983, 991, 997, 1009
]


def test_up_to():
    from data_mining.primes import up_to
    assert_equal(up_to(0), [])
    assert_equal(up_to(1), [1])
    assert_equal(up_to(7), primes_up_to_1009[:5])
    assert_equal(up_to(8), primes_up_to_1009[:5])
    assert_equal(up_to(1009), primes_up_to_1009)
    assert_equal(up_to(1010), primes_up_to_1009)


def test_first():
    from data_mining.primes import first
    assert_equal(first(0), [])
    assert_equal(first(1), [1])
    assert_equal(first(5), primes_up_to_1009[:5])
    assert_equal(first(8), primes_up_to_1009[:8])
    assert_equal(first(len(primes_up_to_1009)), primes_up_to_1009)
