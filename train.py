import torch
import random
import json
import zipfile
import numpy as np
from collections import Counter
from model import SkipGramNegSampling

def train():
    EMBED_DIM = 200
    WINDOW_SIZE = 2
    NEG_SAMPLES = 5
    BATCH_SIZE = 512
    EPOCHS = 1
    LR = 0.003
    
    with open("vocab.json") as f:
        word2id = json.load(f)

    VOCAB_SIZE = len(word2id)
    UNK = word2id["<UNK>"]

    word_freq = Counter(word2id.values())
    prob = np.array(list(word_freq.values())) ** 0.75
    prob /= prob.sum()

    model = SkipGramNegSampling(VOCAB_SIZE, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def clean_text(text):
        import re
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    print("Training started...")
    count=100000 
    with zipfile.ZipFile("enwik9.zip") as z:
        with z.open("enwik9") as f:
            for epoch in range(EPOCHS):
                for line in f:
                    tokens = clean_text(line.decode("utf-8", errors="ignore"))
                    ids = [word2id.get(w, UNK) for w in tokens]

                    pairs = []
                    for i, t in enumerate(ids):
                        for j in range(max(0, i-WINDOW_SIZE),
                                    min(len(ids), i+WINDOW_SIZE+1)):
                            if i != j:
                                pairs.append((t, ids[j]))

                    random.shuffle(pairs)

                    for i in range(0, len(pairs), BATCH_SIZE):
                        batch = pairs[i:i+BATCH_SIZE]
                        if not batch:
                            continue

                        target = torch.LongTensor([p[0] for p in batch])
                        context = torch.LongTensor([p[1] for p in batch])
                        negatives = torch.LongTensor(
                            np.random.choice(VOCAB_SIZE,
                                            (len(batch), NEG_SAMPLES),
                                            p=prob)
                        )

                        loss = model(target, context, negatives)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if count==0:
                        break
                    if count%10000==0:
                        print("Remaining lines",count)
                    count-=1

    print("Training complete.")

    embeddings = model.in_embed.weight.detach().cpu().numpy()
    np.save("my_skipgram_embeddings.npy", embeddings)

    print("Embeddings saved as my_skipgram_embeddings.npy")
    with open("word2id.json", "w") as f:
        json.dump(word2id, f)
        