#!/usr/bin/env python
#coding:utf-8

import os

import pickle
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import net

import argparse

def loaddata(filepath):
    with open(filepath,"rb") as of:
        datalist = pickle.load(of)
    return datalist

def calc_average_loss(datalist,model,dropout):
    sum_loss = 0
    for data in datalist:
        x = np.asarray( [ data[2] ] )
        x = Variable(x)
        t = np.asarray( [ data[1] ] )
        t = Variable(t)
        if dropout is True:
            y = model.predictor(x, train = False)
        else:
            y = model.predictor(x)

        loss = F.mean_squared_error(y,t)
        sum_loss += loss.data
    return sum_loss/len(datalist)

def outputting(outputfilepath,data):
    outstr = ""
    for i,loss in enumerate(data):
        outstr += str(i+1) + "," + str(loss) + "\n"
    with open(outputfilepath,"w") as of:
        of.write(outstr)

#入力層の数
input_dim = 10
#中間層のユニット数
n_units1 = 80
n_units2 = 100
n_units3 = 80

pool_size = 10

#dropoutの利用の有無
dropout = True

#データをロード
trainingdatalist = loaddata("../data/training.pk")
testdatalist = loaddata("../data/test.pk")

#モデルを定義
#ネットワークおよび損失関数を定義
model = L.Classifier(net.AutoEncoderLayer5(input_dim, n_units1, n_units2, n_units3,dropout=dropout,pool_size=pool_size),lossfun=F.mean_squared_error)

#勾配法を定義
optimizer = optimizers.Adam()
optimizer.setup(model)

#学習する回数
n_epoch = 10
batchsize = 20

#結果を格納するリスト
trainresults = list()
testresults = list()

for epoch in range(0,n_epoch):
    for i in range(0,len(trainingdatalist),batchsize):
        datalist = trainingdatalist[i:i+batchsize]
    #for data in trainingdatalist:
        #inputと正解データを配列に格納。
        #バッチ処理が前提となっているので1つでも配列にする
        #Variable型でインプットしないといけないので変換も行う
        x = np.asarray( [ data[2] for data in datalist ] )
        x = Variable(x)
        t = np.asarray( [ data[1] for data in datalist ] )
        t = Variable(t)

        #model.predictorで順伝播をする
        if dropout is True:
            y = model.predictor(x,train=True)
        else:
            y = model.predictor(x)

        #その順伝播させたものと正解データと損失関数に基づき
        #重みを更新する
        optimizer.update(model.lossfun,y,t)

    #1回学習を終えるごとにロス値を算出し結果に格納
    train_loss = calc_average_loss(trainingdatalist,model,dropout)
    test_loss = calc_average_loss(testdatalist,model,dropout)
    trainresults.append(train_loss)
    testresults.append(test_loss)
    print("epoch num: " + str(epoch+1))
    print("train loss: " + str(train_loss))
    print("test loss: " + str(test_loss))

n_units1 = 80
n_units2 = 100
n_units3 = 80

outputfilename = "autoencoder_training_" + "unit1_" + str(n_units1) + "_" + "units2_" + str(n_units2) + "_" + "units3_" + str(n_units3) + "dropout_" + str(dropout).lower() + ".txt"

outputfilepath = os.path.join("..","result_csv",outputfilename)
outputting(outputfilepath,trainresults)

outputfilename = "autoencoder_test_" + "unit1_" + str(n_units1) + "_" + "units2?" + str(n_units2) + "_" + "units3_" + str(n_units3) + "dropout_" + str(dropout).lower() + ".txt"
outputfilepath = os.path.join("..","result_csv",outputfilename)
outputting(outputfilepath,testresults)
