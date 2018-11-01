from __future__ import print_function, division


def up_to(n):
    """Returns prime numbers <= n.

    Source: Based on:
        http://code.activestate.com/recipes/578923-first-n-primes-numbers/

    Args:
        n: the highest prime to return

    Returns:


    """
    prime_candidates = range(0, n + 1)
    kmax = int(n ** 0.5) + 1
    for k in range(2, kmax):
        non_primes = range(k, n + 1, k)
        del non_primes[0]
        for i in non_primes:
            prime_candidates[i] = 0

    return filter(lambda x: x != 0, prime_candidates)


def first(n):
    """Returns first n prime numbers

    Source: Based on:
        https://www.daniweb.com/programming/software-development/threads/233730/to-find-first-n-prime-numbers

    Args:
        n:

    Returns:

    """
    if n > 10000:
        raise ValueError("n must be smaller than 10,000")
    if n == 0:
        return []
    ten_thousandth_prime = 104729
    primes = []
    # Loop through 9999 possible prime numbers
    for a in xrange(1, ten_thousandth_prime):
        # Loop through every number it could divide by
        for b in range(2, a):
            # Does b divide evenly into a ?
            if a % b == 0:
                break
        # Loop exited without breaking ? (It is prime)
        else:
            # Add the prime number to our list
            primes.append(a)
        # We have enough to stop ?
        if len(primes) == n:
            return primes
