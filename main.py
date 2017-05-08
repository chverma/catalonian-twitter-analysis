#!/usr/bin/env python
import preprocess
from models.SVM import SVM
import argparse
import numpy as np
import random

def to_label(line):
    stance = line.split(':::')[1]
    return 0 if stance == 'AGAINST' else 1 if stance == 'NEUTRAL' else 2

def to_identifier(line):
    stance = line.split(':::')[0]

STANCE = ['AGAINST', 'NEUTRAL', 'FAVOR']
'''
The evaluation will be performed according to standard metrics.
In particular, to evaluate stance we will use the macro-average
of F-score (FAVOR) and F-score (AGAINST) as evaluation metric
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-trdata', metavar='Training', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-trlabels', metavar='Labels', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-tedata', metavar='Test', type=str, nargs=1,
                        help='path to the test file', required=True)
    parser.add_argument('-load', metavar='Loadfile', type=str,
                        help='load svm model', default=None)
    parser.add_argument('-save', metavar='Savefile', type=str,
                        help='save svm model', default=None)
    parser.add_argument('-fout', metavar='save output', type=str,
                        help='save output', default=None)
    args = parser.parse_args()
    train_file = args.trdata[0]
    test_file = args.tedata[0]
    train_labels_file = args.trlabels[0]
    load = args.load
    save = args.save
    fout = args.fout
    print('open files...')
    train = open(train_file, 'r', encoding='utf8').readlines()
    train_labels = open(train_labels_file, 'r', encoding='utf8').readlines()
    test = open(test_file, 'r', encoding='utf8').readlines()

    print('training bag of words...')
    bag_of_words = preprocess.train_bag_of_words(train)

    print('computing bags of words...')
    tr_data = np.array(list(map(bag_of_words, train)), dtype=np.float32)
    te_data = np.array(list(map(bag_of_words, test)), dtype=np.float32)
    te_identifier = [list(map(to_identifier, test))]
    print(tr_data.shape, type(tr_data))
    print(te_data.shape, type(te_data))
    print('computing training labels...')
    tr_labels = np.array(list(map(to_label, train_labels)), dtype=np.int32)

    svm = SVM()
    shuffledInd = np.arange(0, len(tr_data))

    ind = int(len(tr_data)*.8)
    random.shuffle(shuffledInd)

    if load is not None:
        print('loading SVM...')
        svm.load(load)
        shuffledInd = np.load('shuffledInd-'+load+".npy")
        tr_data = tr_data[shuffledInd]
        tr_labels = tr_labels[shuffledInd]
    else:
        print('training SVM...')
        tr_data = tr_data[shuffledInd]
        tr_labels = tr_labels[shuffledInd]
        svm.train(tr_data[:ind], tr_labels[:ind])

    if save is not None:
        np.save('shuffledInd-'+save+".npy", shuffledInd)
        svm.save(save)

    responses = svm.predict(te_data)
    if fout is not None:
        fileOut = open(fout, '+w')
        for r, i in zip(responses, te_identifier):
            fileOut.write('%s,%s,DUMMY' % (i, STANCE[r]))
        fileOut.close()
    svm.evaluate(tr_data[ind:], tr_labels[ind:])
    print('finished.')
