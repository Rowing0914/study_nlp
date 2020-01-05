import re
import time
import numpy as np

from scipy import spatial
from multiprocessing import Pool
from nltk.corpus import gutenberg
from gensim.models import Word2Vec


sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))

print('Type of corpus: ', type(sentences))
print('Length of corpus: ', len(sentences))

print("=== Raw Text Data ===")
print(sentences[0])
print(sentences[1])
print(sentences[10])


""" Preprocess data
- Use re module to preprocess data
- Convert all letters into lowercase
- Remove punctuations, numbers, etc
"""
for i in range(len(sentences)):
	sentences[i] = [
	word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)
	]

print("=== Processed Text Data ===")
print(sentences[0])
print(sentences[1])
print(sentences[10])

model = Word2Vec(sentences=sentences,
				 size=100,
				 sg=1,
				 window=3,
				 min_count=1,
				 iter=10,
				 workers=Pool()._processes)

begin = time.time()
model.init_sims(replace = True)
model.save('word2vec_model')
print("Training of word2vec took: {:.4f}".format(time.time() - begin))

model = Word2Vec.load('word2vec_model')

print("=== After Training ===")
print(model.most_similar('hamlet'))

def cosine_similarity(v1, v2):
	return 1 - spatial.distance.cosine(v1, v2)

print("cosine similarity: {}".format(cosine_similarity(v1=model['king'], v2=model['queen'])))