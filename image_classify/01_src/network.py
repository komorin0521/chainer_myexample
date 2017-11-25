#!/usr/bin/env python3
import chainer
import chainer.links as L
import chainer.functions as F


class LinearNet(chainer.Chain):
    def __init__(self, n_units_h1, n_units_h2, n_out):
        super(LinearNet, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units_h1)
            self.l2 = L.Linear(None, n_units_h2)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def extract_hidden(self, x):
        h1 = F.relu(self.l1(x))
        return F.relu(self.l2(h1))


