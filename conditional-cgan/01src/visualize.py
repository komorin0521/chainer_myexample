#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst, image_ch):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols

        # add from example
        indexlist = []
        for i in range(rows):
            for j in range(cols):
                indexlist.append(i)
        label_index = np.array(indexlist, dtype=np.int32)
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images, label_index)))
        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)

        no = 0
        for data in x:
            no += 1
            # array -> PIL format
            data = data.transpose(1, 2, 0)
            if image_ch == 1:
                h, w, _ = data.shape
                data = data.reshape((h, w))
            preview_dir = '%s/preview/%s' % (dst, trainer.updater.iteration)
            preview_path = preview_dir +\
                '/image{:0>8}.png'.format(no)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(data).save(preview_path)
    return make_image
