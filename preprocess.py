import nltk
import argparse

tokenize = lambda text: nltk.tokenize.word_tokenize(text, language='spanish')
stop_punkt = '.,:;¡!¿?#-@http'


def get_contents(csv):
    return (line.split(',', maxsplit=1)[-1] for line in csv)


def train_bag_of_words(csv):
    d = {}
    k = 1
    for message in get_contents(csv):
        for token in tokenize(message):
            if token not in stop_punkt:
                d[token] = k
                k += 1

    def bag_of_words(message):
        representation = [0] * (k + 1)
        for token in tokenize(message):
            if token not in stop_punkt:
                representation[d.get(token, 0)] += 1
        return representation

    return bag_of_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', metavar='TRAIN', type=str, nargs=1,
                        help='path to the train file', required=True)
    parser.add_argument('-test', metavar='TEST', type=str, nargs=1,
                        help='path to the test file', required=True)
    args = parser.parse_args()
    train_file = args.train[0]
    test_file = args.test[0]
    
    train = open(train_file, 'r', encoding='utf8').readlines()
    test = open(test_file, 'r', encoding='utf8').readlines()
    
    bag_of_words = train_bag_of_words(get_contents(train))
    print(list(map(bag_of_words, get_contents(test)))[0])
