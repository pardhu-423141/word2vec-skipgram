import numpy as np
import json

def bias_detection():
    embeddings = np.load("my_skipgram_embeddings.npy")
    with open("word2id.json") as f:
        word2id = json.load(f)

    def get_vec(word):
        return embeddings[word2id[word]]

    gender_pairs = [
        ("he", "she"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother")
    ]

    gender_direction = np.zeros(embeddings.shape[1])

    for w1, w2 in gender_pairs:
        if w1 in word2id and w2 in word2id:
            gender_direction += get_vec(w1) - get_vec(w2)

    gender_direction /= np.linalg.norm(gender_direction)

    def bias_score(word):
        if word not in word2id:
            return None
        v = get_vec(word)
        return np.dot(v, gender_direction)

    occupation_words = [
        "doctor", "nurse", "engineer", "programmer",
        "teacher", "scientist", "artist", "lawyer"
    ]

    print("Bias scores (positive → male, negative → female, zero → neutral)\n")

    for w in range(len(occupation_words)):
        if occupation_words[w] in word2id:
            occupation_words[w]=[occupation_words[w],bias_score(occupation_words[w])]
            
    return occupation_words