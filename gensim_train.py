from gensim.models import Word2Vec
import zipfile
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

def gen_train():
    sentences = []

    with zipfile.ZipFile("enwik9.zip") as z:
        with z.open("enwik9") as f:
            for i, line in enumerate(f):
                tokens = clean_text(line.decode("utf-8", errors="ignore"))
                if len(tokens) > 1:
                    sentences.append(tokens)
                if i > 200000:   # limit for faster training
                    break

    print("Training Gensim Word2Vec...")
    w2v = Word2Vec(
        sentences,
        vector_size=200,
        window=2,
        sg=1,              
        negative=5,
        min_count=5,
        workers=4
    )

    w2v.save("gensim_word2vec.model")
    print("Gensim model saved.")
