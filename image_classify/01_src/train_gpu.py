#!/usr/bin/env python
#coding:utf-8

import pickle
import time
import sys
import os

import numpy as np

from PIL import Image

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import net


def loaddata(inputfilepath):
    with open(inputfilepath,"r") as rf:
        inputdata = rf.readlines()
    inputdata = [ (data.strip()).split(",") for data in inputdata ]
    return inputdata

def splitdata(inputdata):
    labellist = [ "a", "i", "u", "e", "o" ]

    trainingdata = dict()
    testdata = dict()
    for label in labellist:
        trainingdata[label] = list()
        testdata[label] = list()

    for label, outputfilepath in inputdata:
        if len(trainingdata[label]) < 15:
            trainingdata[label].append(outputfilepath)
        else:
            testdata[label].append(outputfilepath)
    return trainingdata, testdata

def convert_data_to_variable_type(inputdata):
    labellist = [ "a", "i", "u", "e", "o" ]
    labeldict = dict()
    for i, label in enumerate(labellist):
        labeldict[label] = i
    outputlist = list()

    for label in inputdata:
        inputfilepathlist = inputdata[label]
        for inputfilepath in inputfilepathlist:
            inputimage = np.array(Image.open(inputfilepath))
            imgdata = inputimage.reshape(32*32)
            imgdata = np.array(inputimage,dtype=np.float32)
            outputlist.append( [ label, imgdata  ,labeldict[label], inputfilepath ] )
    return outputlist

def calc_accuracy(datalist, model):
    print("======== calc accuracy ============")
    model.predictor.train = False
    rightnum = 0
    x = cuda.cupy.asarray( [ data[1] for data in datalist ] )
    x = Variable(x)
    y = model.predictor(x)
    t = cuda.cupy.asarray( [ data[2] for data in datalist ], dtype=np.int32 )
    t = Variable(t)
    loss = model(x,t)
    return model.loss.data, model.accuracy.data

def outputting(outputdata, outputfilepath):
    with open(outputfilepath, "w") as of:
         of.write("\n".join(outputdata))

inputfilepath = "../02_inputdata/labels.txt"
inputdata = loaddata(inputfilepath)

traindata, testdata = splitdata(inputdata)

trainlist = convert_data_to_variable_type(traindata)
testlist = convert_data_to_variable_type(testdata)

n_units = 1000



#モデルを定義
#ネットワークおよび損失関数を定義
model = L.Classifier(net.ImageClassifyModel(32*32,n_units,5))
cuda.get_device(0).use()
model.to_gpu()

#勾配法を定義
optimizer = optimizers.Adam()
optimizer.setup(model)

batchsize = 5

np.random.seed(0)

np.random.shuffle(trainlist)
np.random.shuffle(testlist)

epoch_num = 50

results = { 'train' : dict(), 'test' : dict() }

for k in results:
    results[k]['loss'] = list()
    results[k]['accuracy'] = list()

for epoch in range(1,epoch_num+1):
    for i in range(0,len(trainlist),batchsize):
    #for data in trainlist:
        x = cuda.cupy.asarray( [ data[1] for data in trainlist[i:i+batchsize] ] )
        x = Variable(x)
        t = cuda.cupy.asarray( [ data[2] for data in trainlist[i:i+batchsize] ], dtype=np.int32 )
        t = Variable(t)
        model.predictor.train = True
        optimizer.update(model, x, t)

    #1回学習を終えるごとにロス値を算出し結果に格納
    train_loss, train_ac = calc_accuracy(trainlist,model)
    test_loss, test_ac = calc_accuracy(testlist,model)
    print("epoch num: " + str(epoch))
    print("==== loss ====")
    print("train loss:" + str(train_loss))
    print("test loss:" + str(test_loss))

    print("==== accuracy ====")
    print("train accuracy: " + str(train_ac))
    print("test accuracy: " + str(test_ac))

    results['train']['loss'].append(str(epoch) + "," + str(train_loss))
    results['train']['accuracy'].append(str(epoch)+ ',' + str(train_ac))
    results['test']['loss'].append(str(epoch) + ',' + str(test_loss)) 
    results['test']['accuracy'].append(str(epoch) + ',' + str(test_ac))

outputfolder = os.path.join('..', '04_output', '01_csvfiles')

for datatype in results:
    for evaluate_criteria in results[datatype]:
        outputfilename = datatype + '_' + evaluate_criteria + ".txt"
        outputfilepath = os.path.join(outputfolder, outputfilename)
        outputting(results[datatype][evaluate_criteria], outputfilepath)
