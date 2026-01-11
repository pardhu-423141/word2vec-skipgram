import re
import json
import zipfile
import random
from collections import Counter

WIKI_ZIP = "enwik9.zip"
TEXT_FILE = "enwik9"
VOCAB_FILE = "vocab.json"
WINDOW_SIZE = 2

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

print("Loading vocabulary...")
with open(VOCAB_FILE, "r") as f:
    word2id = json.load(f)

UNK = word2id["<UNK>"]

def generate_skipgram_pairs():
    pairs = []

    with zipfile.ZipFile(WIKI_ZIP) as z:
        with z.open(TEXT_FILE) as f:
            for line in f:
                tokens = clean_text(line.decode("utf-8", errors="ignore"))
                ids = [word2id.get(w, UNK) for w in tokens]

                for i, target in enumerate(ids):
                    start = max(0, i - WINDOW_SIZE)
                    end = min(len(ids), i + WINDOW_SIZE + 1)

                    for j in range(start, end):
                        if i != j:
                            pairs.append((target, ids[j]))

    return pairs
