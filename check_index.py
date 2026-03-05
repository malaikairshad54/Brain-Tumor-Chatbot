import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    print("Loading embedding model (this may download weights)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    index = faiss.read_index("brain_index.faiss")

    with open("texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)

    query = "What is a brain tumor?"
    print(f"Searching for: {query}")

    emb = model.encode([query])
    emb = np.array(emb).astype("float32")

    distances, indices = index.search(emb, 3)

    print("Top results:")
    for rank, idx in enumerate(indices[0]):
        print("----")
        print(f"Rank {rank+1} — index {idx} — distance {distances[0][rank]:.4f}")
        print(texts[idx])

if __name__ == '__main__':
    main()
