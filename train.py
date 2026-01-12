import torch
import json
import zipfile
import numpy as np
from model import SkipGramNegSampling
import re

def train():
    # ---------------- CONFIG ----------------
    EMBED_DIM = 200
    WINDOW_SIZE = 2
    NEG_SAMPLES = 5
    BATCH_SIZE = 512
    EPOCHS = 1
    LR = 0.003
    MAX_LINES = 1000000   # limit training for speed
    # ----------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    with open("vocab.json") as f:
        word2id = json.load(f)

    VOCAB_SIZE = len(word2id)
    UNK = word2id["<UNK>"]

    # Negative sampling distribution (uniform fallback)
    prob = torch.ones(VOCAB_SIZE, device=device)
    prob = prob.pow(0.75)
    prob /= prob.sum()

    # Model
    model = SkipGramNegSampling(VOCAB_SIZE, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return text.split()

    print("Training started...")

    with zipfile.ZipFile("enwik9.zip") as z:
        with z.open("enwik9") as f:

            line_count = 0
            targets, contexts = [], []

            for line in f:
                if line_count >= MAX_LINES:
                    break

                tokens = clean_text(line.decode("utf-8", errors="ignore"))
                ids = [word2id.get(w, UNK) for w in tokens]

                for i, center in enumerate(ids):
                    start = max(0, i - WINDOW_SIZE)
                    end = min(len(ids), i + WINDOW_SIZE + 1)

                    for j in range(start, end):
                        if i != j:
                            targets.append(center)
                            contexts.append(ids[j])

                        if len(targets) == BATCH_SIZE:
                            target = torch.tensor(targets, device=device)
                            context = torch.tensor(contexts, device=device)

                            negatives = torch.multinomial(
                                prob,
                                len(target) * NEG_SAMPLES,
                                replacement=True
                            ).view(len(target), NEG_SAMPLES)

                            loss = model(target, context, negatives)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            targets.clear()
                            contexts.clear()

                line_count += 1
                if line_count % 10_000 == 0:
                    print(f"Processed {line_count} lines")

    print("Training complete.")

    embeddings = model.in_embed.weight.detach().cpu().numpy()
    np.save("my_skipgram_embeddings.npy", embeddings)

    print("Embeddings saved as my_skipgram_embeddings.npy")

    with open("word2id.json", "w") as f:
        json.dump(word2id, f)
