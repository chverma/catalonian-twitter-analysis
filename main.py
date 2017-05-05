#!/usr/bin/env python
import preprocess
from models.SVM import SVM
import argparse
import numpy as np


def to_label(line):
    stance = line.split(':::')[1]
    return 0 if stance == 'AGAINST' else 1 if stance == 'NEUTRAL' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-trdata', metavar='Training', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-trlabels', metavar='Labels', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-tedata', metavar='Test', type=str, nargs=1,
                        help='path to the test file', required=True)
    parser.add_argument('-load', metavar='Loadfile', type=str, help='load svm model'
                        , default=None)
    args = parser.parse_args()
    train_file = args.trdata[0]
    test_file = args.tedata[0]
    train_labels_file = args.trlabels[0]
    load = args.load


    print('open files...')
    train = open(train_file, 'r', encoding='utf8').readlines()
    train_labels = open(train_labels_file, 'r', encoding='utf8').readlines()
    test = open(test_file, 'r', encoding='utf8').readlines()

    print('training bag of words...')
    bag_of_words = preprocess.train_bag_of_words(train)

    print('computing bags of words...')
    tr_data = np.array(list(map(bag_of_words, train)), dtype=np.float32)
    te_data = np.array(list(map(bag_of_words, test)), dtype=np.float32)
    print(tr_data.shape, type(tr_data))
    print(te_data.shape, type(te_data))
    print('computing training labels...')
    tr_labels = np.array(list(map(to_label, train_labels)), dtype=np.int32)

    svm = SVM()
    if load is not None:
        print('loading SVM...')
        svm.load(load)
    else:
        print('training SVM...')
        svm.train(tr_data, tr_labels)
        svm.save('prova.xml')

    responses = svm.predict(te_data)
    np.save('respTe.npy', responses)

    #responses = svm.predict(tr_data)
    svm.evaluate(tr_data, tr_labels)
    #np.save('respTr.npy', responses)
    print('finished.')