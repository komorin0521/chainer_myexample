#!/usr/bin/env python3
"""
network definition
"""

import chainer
import chainer.links as L
import chainer.functions as F


class LinearNet(chainer.Chain):
    """
    Linear network
    """
    def __init__(self, n_units_h1, n_units_h2, n_out):
        """
        init network layer
        """
        super(LinearNet, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units_h1)
            self.l2 = L.Linear(None, n_units_h2)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        """
        propagation
        """
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def extract_hidden(self, x):
        """
        extract hidden layer result
        """
        h1 = F.relu(self.l1(x))
        return F.relu(self.l2(h1))
