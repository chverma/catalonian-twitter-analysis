#!/usr/bin/env python
import preprocess
from models.SVM import SVM
import argparse
import numpy as np


def to_label(line):
    stance = line.split(':::')[1]
    return -1 if stance == 'AGAINST' else 0 if stance == 'NEUTRAL' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', metavar='TRAIN', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-train_labels', metavar='TRAIN', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-test', metavar='TEST', type=str, nargs=1,
                        help='path to the test file', required=True)
    args = parser.parse_args()
    train_file = args.train[0]
    test_file = args.test[0]
    train_labels_file = args.train_labels[0]

    print('open files...')
    train = open(train_file, 'r', encoding='utf8').readlines()
    train_labels = open(train_labels_file, 'r', encoding='utf8').readlines()
    test = open(test_file, 'r', encoding='utf8').readlines()

    print('training bag of words...')
    bag_of_words = preprocess.train_bag_of_words(train)

    print('computing bags of words...')
    data = np.array(list(map(bag_of_words, train)), dtype=np.float32)
    print('computing training labels...')
    labels = np.array(list(map(to_label, train_labels)), dtype=np.int32)
    print('training SVM...')
    svm = SVM()
    svm.train(data, labels)
    print('finished.')
