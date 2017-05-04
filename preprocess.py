import nltk

tokenize = nltk.tokenize.word_tokenize


def train_bag_of_words(messages):
    d = {}
    k = 1
    for message in messages:
        for token in tokenize(message):
            if token not in '.,:;¡!¿?':
                d[token] = k
                k += 1

    def bag_of_words(message):
        representation = [0] * (k + 1)
        for token in tokenize(message):
            representation[d.get(token, 0)] += 1
        return representation

    return bag_of_words
