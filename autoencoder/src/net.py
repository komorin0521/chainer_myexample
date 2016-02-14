#!/usr/bin/env python
#encoding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class AutoEncoder(chainer.Chain):
    def __init__(self, n_in, n_units):
        super(AutoEncoder, self).__init__(
            l1=L.Linear(n_in, n_units), #入力層 -> 隠れ層
            l2=L.Linear(n_units,n_in)   #隠れ層 -> 出力層
        )

    def __call__(self, x):
        # 順伝播を行う関数
        h1 = F.sigmoid(self.l1(x)) #入力層 -> 隠れ層へ伝播
        return self.l2(h1)        #隠れ層 -> 出力層へ伝播
