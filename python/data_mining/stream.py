from __future__ import print_function, division

import abc
import numpy as np


class StreamProcessor(object):
    """Base class for stream processors"""

    def __call__(self, items):
        """Processed the whole stream of items.

        Args:
            items (Iterable(object)) the stream of items to process.
        """
        for item in items:
            self.put(item)

    @abc.abstractmethod
    def put(self, item):
        """The method for processing one item"""
        raise NotImplementedError('')

    @abc.abstractmethod
    def reset(self):
        """Resets the stream processor"""
        raise NotImplementedError('')


class ReservoirSampling(StreamProcessor):
    """Maintains a UNIFORM SAMPLE of processed items up to any time t."""
    def __init__(self, size, seed=None):
        self.size = size
        self.t = None
        self.reservoir = None
        self.seed = seed
        np.random.seed(seed)
        self.reset()

    def put(self, item):
        self.t += 1
        if len(self.reservoir) < self.size:
            self.reservoir.append(item)
        else:
            replace_probability = self.size / self.t
            if np.random.random() < replace_probability:
                replace_idx = np.random.randint(0, self.size)
                self.reservoir[replace_idx] = item

    def reset(self):
        self.reservoir = []
        self.t = 0
