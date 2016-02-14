#!/usr/bin/env python
#coding:utf-8

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

def loaddata(filepath):
    with open(filepath,"rb") as of:
        datalist = pickle.load(of)
    return datalist

def calc_average_loss(datalist,model):
    sum_loss = 0
    for data in datalist:
        x = np.asarray( [ data[2] ] )
        x = Variable(x)
        t = np.asarray( [ data[1] ] )
        t = Variable(t)
        loss = F.mean_squared_error(model.predictor(x),t)
        sum_loss += loss.data
    return sum_loss/len(datalist)

#入力層の数
input_dim = 9
#中間層のユニット数
n_units = 5

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
n_epoch = 100

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
plt.plot([ i+1 for i in range(0,len(testresults)) ], testresults, label="test" )
plt.xlabel("epoch num")
plt.ylabel("squared loss")
plt.legend()
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
 
