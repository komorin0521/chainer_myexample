#!/usr/bin/env python
#coding:utf-8

import pickle
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import net

def convert_data_to_variable_type(inputfilepath):
    inputimage = np.array(Image.open(inputfilepath))
    imgdata = inputimage.reshape(32*32)
    imgdata = np.array(inputimage,dtype=np.float32)
    return imgdata


def load_model(filepath):
    with open(filepath, "rb") as rf:
        model = pickle.load(rf)
    return model

def predict(model, inputdata):
    x = Variable(np.asarray( [ inputdata ] ))
    y = model.predictor(x)
    y = F.softmax(y)

    p_max = None
    index = None
    for i, p in enumerate(y.data[0]):
        if (p_max is None) or (p_max < p):
            p_max = p
            index = i
    return index, p_max

def evaluate(p, index, labellist, tth=0.5):
    if p >= tth:
        label = labellist[index]
    else:
        label = "Nothing"
    return label

modelpath = "../05_model/model_cpu.pk"
model = load_model(modelpath)


inputimagefilepath = "../02_inputdata/03_u/02_monochrome/04_u.png"
inputdata = convert_data_to_variable_type(inputimagefilepath)

index, p = predict(model,inputdata)

labellist = [ "a", "i", "u", "e", "o" ]

label = evaluate(p, index,labellist)

print(label)
