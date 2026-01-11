import re
import json
import requests
import zipfile
import os
from collections import Counter

WIKI_ZIP_URL = "https://mattmahoney.net/dc/enwik9.zip"
ZIP_FILE = "enwik9.zip"
TEXT_FILE = "enwik9"
VOCAB_SIZE = 50000
VOCAB_FILE = "vocab.json"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

if not os.path.exists(ZIP_FILE):
    print("Downloading Wikipedia dump safely...")
    with requests.get(WIKI_ZIP_URL, stream=True) as r:
        r.raise_for_status()
        with open(ZIP_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print("Download complete.")
else:
    print("Zip file already exists.")

print("Processing Wikipedia text...")
word_counts = Counter()

with zipfile.ZipFile(ZIP_FILE) as z:
    with z.open(TEXT_FILE) as wiki_file:
        for i, line in enumerate(wiki_file):
            tokens = clean_text(line.decode("utf-8", errors="ignore"))
            word_counts.update(tokens)

            if (i + 1) % 100000 == 0:
                print(f"Lines processed: {i+1}")

most_common = word_counts.most_common(VOCAB_SIZE - 1)
vocab = {"<UNK>": 0}

for i, (word, _) in enumerate(most_common, start=1):
    vocab[word] = i

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f)

print("Vocabulary saved.")
