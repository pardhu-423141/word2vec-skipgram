import numpy as np
import json
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def compare():
    my_embeddings = np.load("my_skipgram_embeddings.npy")
    with open("word2id.json") as f:
        word2id = json.load(f)

    id2word = {i:w for w,i in word2id.items()}

    gensim_model = Word2Vec.load("gensim_word2vec.model")
    gensim_vectors = gensim_model.wv

    def compare_word(word):
        if word not in word2id or word not in gensim_vectors:
            return None

        v1 = my_embeddings[word2id[word]]
        v2 = gensim_vectors[word]

        sim = cosine_similarity([v1], [v2])[0][0]
        return sim

    words = ["king", "queen", "man", "woman", "computer", "science"]

    for w in range(len(words)):
        words[w]=[words[w],compare_word(words[w])]
        print(f"{words[w][0]:10s} → cosine similarity = {words[w][1]}")
    return words