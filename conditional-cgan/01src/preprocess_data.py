#!/usr/bin/env python2
# coding: utf-8

import argparse
import os

def importingargs():
    """
    definition of arguments
    :return:
    """
    parser = argparse.ArgumentParser("Prepare dataset")
    parser.add_argument("--dataset-filepath", "-d", help="")
    parser.add_argument("--labelfilepath", "-l", help="")
    parser.add_argument("--outputfilepath", "-o", help="")
    args = parser.parse_args()
    return args.dataset_filepath, args.labelfilepath, args.outputfilepath

def get_label(inputs):
    """
    Getting label from inputs
    Please modify according to your data
    This example is getting label from filename, 
    the format is "no_{label}.png"
    """
    return inputs.split(".")[0].split("_")[1]

def main():
    """
    main
    """

    dataset_filepath, labelfilepath, outputfilepath = importingargs()

    labeldict = {}
    with open(labelfilepath, "r") as rf:
        labeldict = { line.strip().split()[1]:line.strip().split()[0] for line in rf.readlines() }


    filenamelist = os.listdir(dataset_filepath)

    outputlist = []
    for filename in filenamelist:
        label = get_label(filename)
        outputlist.append(" ".join([filename, labeldict[label]]))

    with open(outputfilepath, "w") as of:
        of.write(("\n".join(outputlist)))


if __name__ == "__main__":
    main()


# Local Variables:
# coding: utf-8
# End:
