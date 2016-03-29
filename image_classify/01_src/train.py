#!/usr/bin/env python
#coding:utf-8

import pickle
import time
import sys

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
        print(label)
        inputfilepathlist = inputdata[label]
        for inputfilepath in inputfilepathlist:
            inputimage = np.array(Image.open(inputfilepath))
            imgdata = inputimage.reshape(64*64)
            imgdata = np.array(inputimage,dtype=np.float32)
            outputlist.append( [ label, imgdata  ,labeldict[label] ] )
    return outputlist

def calc_accuracy(datalist, model):
    rightnum = 0
    x = np.asarray( [ data[1] for data in datalist ] )
    x = Variable(x)
    y = model.predictor(x)
    t = np.asarray( [ data[2] for data in datalist ], dtype=np.int32 )
    t = Variable(t)
    accuracy = F.accuracy(y, t)
    return accuracy.data
    #for data in datalist:
    #    x = np.asarray( [ data[1] ] )
    #    x = Variable(x)
    #    t = np.asarray( [ data[2] ], dtype=np.int32 )
    #    t = Variable(t)
    #    y = F.softmax(model.predictor(x))
    #    a = F.accuracy(y,t)
    #    rightnum += a.data
    #return rightnum/len(datalist)


inputfilepath = "../02_inputdata/labels.txt"
inputdata = loaddata(inputfilepath)

traindata, testdata = splitdata(inputdata)

trainlist = convert_data_to_variable_type(traindata)
testlist = convert_data_to_variable_type(testdata)

n_units = 10000

#モデルを定義
#ネットワークおよび損失関数を定義
model = L.Classifier(net.MyNetwork(64*64,n_units,5), lossfun=F.softmax_cross_entropy)

#勾配法を定義
optimizer = optimizers.Adam()
optimizer.setup(model)

"""
for epoch in range(0,20):
    for data in trainlist:
        x = np.asarray( [ data[1] ] )
        x = Variable(x)
        t = np.asarray( [ data[2] ], dtype=np.int32 )
        t = Variable(t)
        optimizer.update(model.lossfun, model.predictor(x), t)

    #1回学習を終えるごとにロス値を算出し結果に格納
    train_loss = calc_accuracy(trainlist,model)
    test_loss = calc_accuracy(testlist,model)
    print("epoch num: " + str(epoch+1))
    print("train loss: " + str(train_loss))
    print("test loss: " + str(test_loss))
"""

batchsize = 5

np.random.shuffle(trainlist)
np.random.shuffle(testlist)

for epoch in range(0,20):
    for i in range(0,len(trainlist),batchsize):
    #for data in trainlist:
        x = np.asarray( [ data[1] for data in trainlist[i:i+batchsize] ] )
        x = Variable(x)
        t = np.asarray( [ data[2] for data in trainlist[i:i+batchsize] ], dtype=np.int32 )
        t = Variable(t)
        optimizer.update(model.lossfun, model.predictor(x), t)

    #1回学習を終えるごとにロス値を算出し結果に格納
    train_loss = calc_accuracy(trainlist,model)
    test_loss = calc_accuracy(testlist,model)
    print("epoch num: " + str(epoch+1))
    print("train loss: " + str(train_loss))
    print("test loss: " + str(test_loss))




"""
#入力層の数
input_dim = 10
#中間層のユニット数
n_units = 8

#データをロード
trainingdatalist = loaddata("../data/training.pk")
testdatalist = loaddata("../data/test.pk")

#モデルを定義
#ネットワークおよび損失関数を定義
model = L.Classifier(net.AutoEncoder(input_dim,n_units),lossfun=F.mean_squared_error)

#勾配法を定義
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習する回数
n_epoch = 10

#結果を格納するリスト
trainresults = list()
testresults = list()

for epoch in range(0,n_epoch):
    for data in trainingdatalist:
        #inputと正解データを配列に格納。
        #バッチ処理が前提となっているので1つでも配列にする
        #Variable型でインプットしないといけないので変換も行う
        x = np.asarray( [ data[2] ] )
        x = Variable(x)
        t = np.asarray( [ data[1] ] )
        t = Variable(t)

        #model.predictorで順伝播をする
        #その順伝播させたものと正解データと損失関数に基づき
        #重みを更新する
        optimizer.update(model.lossfun,model.predictor(x),t)

    #1回学習を終えるごとにロス値を算出し結果に格納
    train_loss = calc_average_loss(trainingdatalist,model)
    test_loss = calc_average_loss(testdatalist,model)
    trainresults.append(train_loss)
    testresults.append(test_loss)
    print("epoch num: " + str(epoch+1))
    print("train loss: " + str(train_loss))
    print("test loss: " + str(test_loss))


#結果をもとにグラフを作成
plt.plot([ i+1 for i in range(0,len(trainresults)) ], trainresults,label="training")
plt.plot([ i+1 for i in range(0,len(testresults)) ], testresults, label="test")
plt.xlabel("epoch num",fontsize=24)
plt.ylabel("squared loss",fontsize=24)
plt.legend(fontsize=24)
plt.savefig("../result/result.png")


#何個か取り出してインプットと順伝播させたものを表示させてみる
for data in testdatalist[:5]:
    x = np.asarray( [ data[2] ] )
    x = Variable(x)
    t = np.asarray( [ data[1] ] )
    t = Variable(t)
    y = model.predictor(x)

    print("=============================")
    print("original data:")
    print(t.data)

    print("input data:")
    print(x.data)

    print("predict data")
    print(y.data)
    print("=============================")

print("Finished")
print(time.ctime())
"""
