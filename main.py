#!/usr/bin/python3.6
import preprocess
from models.SVM import SVM
import argparse


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

    train = open(train_file, 'r', encoding='utf8').readlines()
    train_labels = open(train_labels_file, 'r', encoding='utf8').readlines()
    test = open(test_file, 'r', encoding='utf8').readlines()

    bag_of_words = preprocess.train_bag_of_words(train)

    data = list(map(bag_of_words, train))
    svm = SVM()
    svm.train(data, map(lambda l: l.split(':::')[1], train_labels))
