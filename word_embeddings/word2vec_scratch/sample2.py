import numpy as np
from collections import defaultdict


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Network(object):
    def __init__(self, word_cnt=5, dim_embed=5, lr=0.001):
        self.w = np.random.uniform(low=-0.8, high=0.8, size=(word_cnt, dim_embed))
        self.v = np.random.uniform(low=-0.8, high=0.8, size=(dim_embed, word_cnt))
        self._lr = lr

    def forward(self, x):
        h = np.dot(self.w.T, x)
        o = np.dot(self.v.T, h)
        y_c = softmax(x=o)
        return y_c, h, o

    def backward(self, error, hidden, x):
        dL_dV = np.outer(hidden, error)
        dL_dW = np.outer(x, np.dot(self.v, error))

        self.w -= self._lr * dL_dW
        self.v -= self._lr * dL_dV


def _test_network():
    sample_word = np.array([[1, 0, 0]]).T
    target = np.array([[0, 0, 1]]).T
    network = Network(word_cnt=sample_word.shape[0])
    pred, h, o = network.forward(x=sample_word)
    error = np.abs(target - pred)
    network.backward(error=error, hidden=h, x=sample_word)


class Word2Vec(object):
    def __init__(self, corpus, dim_word=100, lr=0.01, epochs=10, window_size=2):
        self._dim_word = dim_word
        self._lr = lr
        self._epochs = epochs
        self._window_size = window_size

        self._preparation(corpus=corpus)
        self._train(corpus=corpus)

    def _preparation(self, corpus):
        """ Prep for the training """
        # count the occurrence of each word
        self.word_cnt = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                self.word_cnt[word] += 1
        self.words_list = sorted(list(self.word_cnt.keys()), reverse=False)

        # create master tables(word2ind, ind2word)
        self.word2ind, self.ind2word = dict(), dict()
        for i, word in enumerate(self.words_list):
            self.word2ind[word] = i
            self.ind2word[i] = word

        # instantiate the network and identity matrix
        self.word_onehot = np.eye(len(self.words_list))
        self.network = Network(word_cnt=len(self.words_list),
                               dim_embed=self._dim_word)

    def _train(self, corpus):
        for sentence in corpus:
            for i, word in enumerate(sentence):
                # prep for the training data
                context_words = list()
                for j in range(i - self._window_size, i + self._window_size + 1):
                    if j != i and len(sentence) - 1 >= j >= 0:
                        word = sentence[j]
                        context_words.append(self.word_onehot[self.word2ind[word]])

                # specify the target/context words
                target = self.word_onehot[self.word2ind[word]]
                context_words = np.asarray(context_words).T

                # forward pass
                y_pred, h, u = self.network.forward(x=target)

                # calculate the error
                y_pred = np.tile(y_pred[:, np.newaxis], (1, context_words.shape[1]))
                error = y_pred - context_words
                error = np.sum(error, axis=-1)

                # backward pass
                self.network.backward(error=error, hidden=h, x=target)


if __name__ == '__main__':
    corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
    word2vec = Word2Vec(corpus=corpus)
