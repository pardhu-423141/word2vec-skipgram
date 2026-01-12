from train import train
from gensim_train import gen_train
from compare_embeddings import compare
from analogy import test_analogies
from bias_detection import bias_detection



if __name__ == "__main__":
    training=True
    if training:
        train()
        gen_train()
    with open("output.txt", "w", encoding="utf-8") as f:

        f.write("Comparing embeddings with gensim embeddings using cosine similarity.\n\n")
        w=compare()
        for i in range(len(w)):
            f.write(f"{w[i][0]:10s} → cosine similarity = {w[i][1]}\n")
        
        f.write("\n\nWord analogy based on trained embeddings.\n\n")
        t=test_analogies()
        for test in t:
            (a,b,c),results=test
            f.write(f"{a}:{b} :: {c}:?\n")
            f.write(str(results)+"\n")
        
        f.write("\n\nDetecting bias in embeddings.\n\n")
        b = bias_detection()
        for i in b:
            f.write(f"{i[0]:12s}: {str(i[1])}\n")
