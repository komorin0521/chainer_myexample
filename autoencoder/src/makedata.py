#!/usr/bin/env python
#enconding:utf-8

import numpy as np
import pickle

np.random.seed(0)

def makerightdata(N):
    """
    [0,1]からなるN bitの配列を生成する関数
    """
    return np.array(np.random.choice(2,N),dtype=np.float32)

def noise(x,N,sigma):
    """
    noise を加える関数
    """
    noisedata = x + np.random.normal(0,sigma,N)
    return np.array(noisedata,dtype=np.float32)



#生成するデータの数
datanum = 1000

#生成データのビット数
N = 10
sigma = 0.25

#基本となるデータの種類を作成
datatypenum = 5
datatypelist = list()
for i in range(0,datatypenum):
    datatypelist.append(makerightdata(N))

for data in  datatypelist:
    print(data)

#ノイズを付加する
datalist = list()
for i in range(0,datanum):
    typenum = np.random.randint(datatypenum)
    t = datatypelist[typenum]
    if i < int(datanum*0.8):
        n = noise(t,N,sigma)
    else:
        n = noise(t,N,sigma+0.1)
    datalist.append( [ len(datalist),t,n ])

with open("../data/training.pk","wb") as of:
    #8割のデータをトレーニングとする
    pickle.dump(datalist[:int(datanum*0.8)],of)

with open("../data/test.pk","wb") as of:
    #2割のデータをテストとする
    pickle.dump(datalist[int(datanum*0.8):],of)
