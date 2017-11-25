#!/usr/bin/env python3
import argparse
import os
import random

# import numpy as np
import chainer
import chainer.links as L

from network import LinearNet
from util import load_data, load_label


def load_my_dataset(filepath, datafolderpath, labelfilepath, test_ratio=0.9):
    """
    loading and make train and test dataests
    filepath: the list of imagefilenamelist
    datafolderpath: the datafolder of inputting image
    labelfilepath: the data of label
    for this example the label is "a", "i", "u", "e", "o"
    """
    labellist = load_label(labelfilepath)

    with open(filepath, "r") as rf:
        filepathlist = [ os.path.join(datafolderpath, line.strip()) for line in rf.readlines() ]

    datasets = list()
    for imagefilepath in filepathlist:
        img = load_data(imagefilepath)
        filename = imagefilepath.split(os.path.sep)[-1]
        label = filename.split(".")[0].split("_")[1]
        index = labellist.index(label)
        datasets.append([img, index])
    random.shuffle(datasets)

    index = int(len(datasets) * test_ratio)
    traindatasets = datasets[:index]
    testdatasets = datasets[index:]

    train_imgs = [ train_data[0] for train_data in traindatasets ]
    train_labels = [ train_data[1] for train_data in traindatasets ]

    test_imgs = [ test_data[0] for test_data in testdatasets ]
    test_labels = [ test_data[1] for test_data in testdatasets ]

    return chainer.datasets.tuple_dataset.TupleDataset(train_imgs, train_labels), chainer.datasets.tuple_dataset.TupleDataset(test_imgs, test_labels)

def importingargs():
    """
    Definition of parser
    """

    parser = argparse.ArgumentParser("The sample training of linear network")
    parser.add_argument("--first-hidden-layer-units", "-fu", type=int, default=500, help="the units num of first hidden")
    parser.add_argument("--second-hidden-layer-units", "-su", type=int, default=250, help="the units num of first hidden")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="batch size in training data")
    parser.add_argument("--label-num", "-l", type=int, default=5)
    parser.add_argument("--epoch", "-e", type=int, default=50, help="epoch num of training")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu device")
    parser.add_argument("--filepath", "-f", help="filepath written the imagefilepath")
    parser.add_argument("--labelpath", "-lf", help="filepath written the def of label")
    parser.add_argument("--datafolderpath", "-df", help="datafolder of imagefile")
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed of random")
    args = parser.parse_args()

    return args.first_hidden_layer_units, args.second_hidden_layer_units, args.batch_size, args.label_num, args.epoch, args.gpu, args.filepath, args.labelpath, args.datafolderpath, args.seed


def main():
    n_units_h1, n_units_h2, batch_size, n_out, epoch_num, gpu, filepath, labelfilepath, datafolderpath, seed = importingargs()

    # set the seed
    random.seed(seed)

    # definition of the network
    model = L.Classifier(LinearNet(n_units_h1, n_units_h2, n_out))

    # gpu setup
    if gpu >= 0:
        print("use gpu")
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    traindata, testdata = load_my_dataset(filepath, datafolderpath, labelfilepath)

    train_iter = chainer.iterators.SerialIterator(traindata, batch_size)
    test_iter = chainer.iterators.SerialIterator(traindata, batch_size, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter,optimizer,device=gpu)
    trainer = chainer.training.Trainer(updater,(epoch_num, 'epoch'), out='result')
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model, device=gpu))

    trainer.extend(chainer.training.extensions.dump_graph('main/loss'))
    trainer.extend(chainer.training.extensions.snapshot(), trigger=(10,'epoch'))
    trainer.extend(chainer.training.extensions.LogReport())

    # save the modelfile
    trainer.extend(chainer.training.extensions.snapshot_object(model, 'model_{.updater.iteration}.npz'), trigger=(10, 'epoch'))
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time' ]))

    # running training
    trainer.run()

if __name__ == "__main__":
    main()
