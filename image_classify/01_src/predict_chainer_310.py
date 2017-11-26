#!/usr/bin/env python3
"""
classify using saved model
"""
import argparse

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

from network import LinearNet
from util import load_data, load_label


def predict(model, inputdata, labellist):
    """
    predict from image data
    """
    x = Variable(np.asarray([inputdata]))
    y = model.predictor(x)
    y = F.softmax(y)

    p_max = None
    index = None
    for i, p in enumerate(y.data[0]):
        if (p_max is None) or (p_max < p):
            p_max = p
            index = i
            label = labellist[index]
    return index, p_max, label


def importingargs():
    """
    Definition of args
    """
    parser = argparse.ArgumentParser("predicting using saved model")
    parser.add_argument("--first-hidden-layer-units", "-fu",
                        type=int, default=500,
                        help="the units num of first hidden")
    parser.add_argument("--second-hidden-layer-units", "-su",
                        type=int, default=100,
                        help="the units num of first hidden")
    parser.add_argument("--label-num", "-l", type=int, default=5)
    parser.add_argument("--modelpath", "-mf", help="model path")
    parser.add_argument("--labelfilepath", "-lf", help="labelfile")
    parser.add_argument("--imagefilepath", "-if",
                        help="imagefilepath you want to predict")
    args = parser.parse_args()
    return args.first_hidden_layer_units,\
            args.second_hidden_layer_units, args.label_num,\
            args.modelpath, args.labelfilepath, args.imagefilepath


def main():
    """
    main
    """
    n_units_h1, n_units_h2, n_out,\
            modelpath, labelfilepath, imagefilepath = importingargs()
    model = L.Classifier(LinearNet(n_units_h1, n_units_h2, n_out))
    chainer.serializers.load_npz(modelpath, model)

    img = load_data(imagefilepath)
    labellist = load_label(labelfilepath)
    index, prob, label = predict(model, img, labellist)
    print("index: %d, prob: %f, label: %s" % (index, prob, label))


if __name__ == "__main__":
    main()
