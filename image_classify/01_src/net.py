#!/usr/bin/env python
#encoding:utf-8

import chainer
import chainer.functions as F
import chainer.links as L


class ImageClassifyModel(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(MyNetwork, self).__init__(
            l1=L.Linear(n_in, n_units),     #入力層 -> 隠れ層
            l2=L.Linear(n_units, n_units),  #隠れ層 -> 隠れ層
            l3=L.Linear(n_units, n_out)     #隠れ層 -> 出力層
        )

    def __call__(self, x):
        # 順伝播を行う関数
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class ImageClssifyModelParallel(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(ImageClssifyModelParallel, self).__init__(
                first0=ImageClassifyModel(n_in, n_units//2, n_units).to_gpu(0),
                first1=ImageClassifyModel(n_in, n_units//2,n_units).to_gpu(1),
                second0=ImageClassifyModel(n_units, n_units//2,n_out).to_gpu(0),
                second1=ImageClassifyModel(n_units, n_units//2, n_out).to_tpu(1))


    def __call__(self, x):
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1a(x1)

        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(1))

        y = y0 + F.copy(y1, 0)

        return y

class ImageClassifyModelwithDropOut(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(ImageClassifyModelwithDropOut, self).__init__(
            l1=L.Linear(n_in, n_units),     #入力層 -> 隠れ層
            l2=L.Linear(n_units, n_units),  #隠れ層 -> 隠れ層
            l3=L.Linear(n_units, n_out)     #隠れ層 -> 出力層
        )
        self.train = None

    def __call__(self, x):
        # 順伝播を行う関数
        h1 = F.dropout(F.relu(self.l1(x)),train=self.train, ratio=0.2)
        h2 = F.dropout(F.relu(self.l2(h1)),train=self.train, ratio=0.5)
        return self.l3(h2)
