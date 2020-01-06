import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from word_embeddings.word2vec_scratch.utils import cosine_similarity, reduce_to_k_dim, plot_embeddings

corpus = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
model = Word2Vec(sentences=corpus, size=5, sg=1, window=2, min_count=1, iter=500)
model.init_sims(replace=True)
print("=== After Training ===")
print(model.most_similar('the'))
print("cosine similarity: {}".format(cosine_similarity(v1=model['quick'], v2=model['jumped'])))

print(model.wv.index2word)
M_reduced = reduce_to_k_dim(M=np.asarray(model.wv[model.wv.index2word]))
word2index = {token: token_index for token_index, token in enumerate(model.wv.index2word)}
plot_embeddings(M_reduced=M_reduced, word2Ind=word2index, words=model.wv.index2word)
plt.savefig("gensim.png")
