import numpy as np
import json
from sklearn import tests
from sklearn.metrics.pairwise import cosine_similarity


def analogy(a, b, c, embeddings, word2id, id2word, top_k=5):
    for w in [a, b, c]:
        if w not in word2id:
            print(f"Word '{w}' not in vocabulary")
            return

    va = embeddings[word2id[a]]
    vb = embeddings[word2id[b]]
    vc = embeddings[word2id[c]]

    target = vb - va + vc

    sims = cosine_similarity([target], embeddings)[0]
    best = np.argsort(sims)[::-1]

    results = []
    for idx in best:
        word = id2word[idx]
        if word not in [a, b, c]:
            results.append((word, sims[idx]))
        if len(results) == top_k:
            break
    return results


def test_analogies():
    tests = [
        ("man", "king", "woman"),
        ("paris", "france", "italy"),
        ("big", "bigger", "small"),
        ("good", "better", "bad")
    ]
    embeddings = np.load("my_skipgram_embeddings.npy")
    with open("word2id.json") as f:
        word2id = json.load(f)

    id2word = {i:w for w,i in word2id.items()}

    for i in range(len(tests)):
        a, b, c = tests[i]
        tests[i]=[tests[i],analogy(a, b, c, embeddings, word2id, id2word)]
    return tests
