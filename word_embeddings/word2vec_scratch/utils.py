import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial
from sklearn.decomposition import PCA


def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def reduce_to_k_dim(M, k=2):
    """ Reduce a matrix of dimensionality
        - size: (num_corpus_words, num_corpus_words) -> (num_corpus_words, k)
    """
    M_reduced = PCA(n_components=k).fit_transform(M)
    return M_reduced


def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.

        Params:
            M_reduced (numpy matrix of shape (num_corpus_words, k)): A k-dim matrix word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """
    for word in words:
        x, y = M_reduced[word2Ind[word]]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)


if __name__ == '__main__':
    M_reduced = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
    word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
    words = ['test1', 'test2', 'test3', 'test4', 'test5']
    plot_embeddings(M_reduced, word2Ind_plot_test, words)
    plt.show()
