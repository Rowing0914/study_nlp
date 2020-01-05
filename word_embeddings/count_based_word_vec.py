"""
Stanford CS224N Assignment 1: Count-Based Word Vectors
This focuses on Count-Based method, doesn't include the Prediction-Based Word Vectors

Full assignment can be found here
    - https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a1_preview/exploring_word_vectors.html
"""

import nltk
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

try:
    from nltk.corpus import reuters
except:
    nltk.download('reuters')
    from nltk.corpus import reuters

np.random.seed(0)
random.seed(0)

START_TOKEN = '<START>'
END_TOKEN = '<END>'


def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]


def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus,
                                            sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = sorted(set([word for sentence in corpus for word in sentence]))
    num_corpus_words = len(corpus_words)
    return corpus_words, num_corpus_words


def check_distinct_words():
    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness.
    # ---------------------

    # Define toy corpus
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    test_corpus_words, num_corpus_words = distinct_words(test_corpus)

    # Correct answers
    ans_test_corpus_words = sorted(
        list(set(["START", "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", "END"])))
    ans_num_corpus_words = len(ans_test_corpus_words)

    # Test correct number of words
    assert (
            num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(
        ans_num_corpus_words, num_corpus_words)

    # Test correct words
    assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(
        str(ans_test_corpus_words), str(test_corpus_words))

    # Print Success
    print("-" * 80)
    print("Passed All Tests!")
    print("-" * 80)


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurrence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words
                given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = np.zeros(shape=(num_words, num_words))
    word2Ind = {word: index for index, word in enumerate(words)}

    # ------------------
    # Write your implementation here.
    for sentence in corpus:
        for idx, word in enumerate(sentence):
            for _idx in range(1, window_size + 1):
                if len(sentence) > idx + _idx:
                    _w = sentence[idx + _idx]
                    cooccur_id = word2Ind[_w]
                    center_id = word2Ind[word]
                    M[cooccur_id, center_id] += 1
                    M[center_id, cooccur_id] += 1

    # ------------------

    return M, word2Ind


def check_compute_co_occurrence_matrix():
    # ---------------------
    # Run this sanity check
    # Note that this is not an exhaustive check for correctness.
    # ---------------------

    # Define toy corpus and get student's co-occurrence matrix
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

    # Correct M and word2Ind
    M_test_ans = np.array(
        [[0., 0., 0., 1., 0., 0., 0., 0., 1., 0., ],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 1., ],
         [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., ],
         [1., 1., 0., 0., 0., 0., 0., 0., 0., 0., ],
         [0., 0., 0., 0., 0., 0., 0., 0., 1., 1., ],
         [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., ],
         [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., ],
         [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., ],
         [1., 0., 0., 0., 1., 1., 0., 0., 0., 1., ],
         [0., 1., 1., 0., 1., 0., 0., 0., 1., 0., ]]
    )
    word2Ind_ans = {'All': 0, "All's": 1, 'END': 2, 'START': 3, 'ends': 4, 'glitters': 5, 'gold': 6, "isn't": 7,
                    'that': 8, 'well': 9}

    # Test correct word2Ind
    assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans,
                                                                                                         word2Ind_test)

    # Test correct M shape
    assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(
        M_test.shape, M_test_ans.shape)

    # Test correct M values
    for w1 in word2Ind_ans.keys():
        idx1 = word2Ind_ans[w1]
        for w2 in word2Ind_ans.keys():
            idx2 = word2Ind_ans[w2]
            student = M_test[idx1, idx2]
            correct = M_test_ans[idx1, idx2]
            if student != correct:
                print("Correct M:")
                print(M_test_ans)
                print("Your M: ")
                print(M_test)
                raise AssertionError(
                    "Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(
                        idx1, idx2, w1, w2, student, correct))

    # Print Success
    print("-" * 80)
    print("Passed All Tests!")
    print("-" * 80)


def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): co-occurrence matrix of word
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10  # Use this parameter in your call to `TruncatedSVD`
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    M_reduced = TruncatedSVD(n_components=k, n_iter=n_iters).fit_transform(M)
    print("Done.")
    return M_reduced


def check_reduce_to_k_dim():
    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness
    # In fact we only check that your M_reduced has the right dimensions.
    # ---------------------

    # Define toy corpus and run student code
    test_corpus = ["START All that glitters isn't gold END".split(" "),
                   "START All's well that ends well END".split(" ")]
    M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
    M_test_reduced = reduce_to_k_dim(M_test, k=2)

    # Test proper dimensions
    assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
    assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

    # Print Success
    print("-" * 80)
    print("Passed All Tests!")
    print("-" * 80)


def plot_embeddings(M_reduced, word2Ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , k)): A k-dim matrix word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """
    for word in words:
        x, y = M_reduced[word2Ind[word]]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word, fontsize=9)


def check_plot_embeddings():
    # ---------------------
    # Run this sanity check
    # Note that this not an exhaustive check for correctness.
    # The plot produced should look like the "test solution plot" depicted below.
    # ---------------------

    print("-" * 80)
    print("Outputted Plot saved as plot_embeddings.png")

    M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
    word2Ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
    words = ['test1', 'test2', 'test3', 'test4', 'test5']
    plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words)
    plt.savefig("plot_embeddings.png")
    plt.clf()
    print("-" * 80)


def final_plot():
    # -----------------------------
    # Run This Cell to Produce Your Plot
    # ------------------------------
    reuters_corpus = read_corpus()
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]  # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words)
    plt.savefig("final_plot.png")
    print("Outputted Plot saved as final_plot.png")
    plt.clf()


if __name__ == '__main__':
    check_distinct_words()
    check_compute_co_occurrence_matrix()
    check_reduce_to_k_dim()
    check_plot_embeddings()
    final_plot()
