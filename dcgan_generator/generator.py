#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

from net import Generator

def make_image(gen, rows, cols, seed, dst):
    np.random.seed(seed)
    n_images = rows * cols
    xp = gen.xp
    z = Variable(xp.asarray(gen.make_hidden(n_images)))
    x = gen(z, test=True)
    x = chainer.cuda.to_cpu(x.data)
    np.random.seed()

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)

    for row in range(0, rows):
        for col in range(0, cols):
            out_array = x[row, :, col]
            out_array = out_array.reshape((H, W, 3))
            no = '{0:04d}'.format(row + col + 1)
            outputfilepath = './%s/image_gen_%s.png' % (dst, no)
            Image.fromarray(out_array).save(outputfilepath)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')

    parser.add_argument('--gen', '-g', help='generator file' )
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--n_hidden', '-n', type=int, default=100,
                        help='Number of hidden units (z)')


    parser.add_argument('--image_num', '-i', type=int, default=100,
                        help='generated image num')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    args = parser.parse_args()

    # Set up a neural network to train
    gen = Generator(n_hidden=args.n_hidden)
    chainer.serializers.load_npz(args.gen, gen)

    make_image( gen, args.image_num, 1, args.seed, args.out)

if __name__ == '__main__':
    main()
