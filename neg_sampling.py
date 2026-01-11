import numpy as np

def negative_sampling_distribution(word_freq):
    prob = np.array(word_freq) ** 0.75
    prob /= prob.sum()
    return prob
