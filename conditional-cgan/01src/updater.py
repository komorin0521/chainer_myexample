#!/usr/bin/env python

from __future__ import print_function

# add
import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
from chainer.training import StandardUpdater
from chainer.dataset import concat_examples

class DCGANUpdater(StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        # add
        self.classnum = kwargs.pop('classnum')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()

        # adding from examples
        # get batch image and label index(not onehot vector)
        x_real, label_index = concat_examples(batch)
        x_real = Variable(self.converter(x_real, self.device)) / 255.

        batchsize = len(batch)
        xp = chainer.cuda.get_array_module(x_real.data)

        gen, dis = self.gen, self.dis

        y_real = dis(x_real)

        z = Variable(xp.asarray(gen.make_hidden(batchsize, label_index)))
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)