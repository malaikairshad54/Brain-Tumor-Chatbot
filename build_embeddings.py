import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Path to your JSON file
DATA_PATH = "data/chunks.json"

if not os.path.exists(DATA_PATH):
    print("ERROR: chunks.json not found inside data/ folder.")
    exit()

print("Loading JSON data...")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract text from each chunk
texts = []

for item in data:
    if "text" in item:
        texts.append(item["text"])

# Load Q&A Data
QA_PATH = "data/questions_answers.json"
if os.path.exists(QA_PATH):
    print(f"Loading Q&A data from {QA_PATH}...")
    with open(QA_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    for item in qa_data:
        # Format: Q: <Question> \n A: <Answer>
        # We use the first pattern as the representative question
        if "patterns" in item and "response" in item:
            question = item["patterns"][0]
            answer = item["response"]
            combined_text = f"Q: {question}\nA: {answer}"
            texts.append(combined_text)
    
    print(f"Added {len(qa_data)} Q&A pairs.")
else:
    print("WARNING: questions_answers.json not found.")

print(f"Total chunks loaded: {len(texts)}")

if len(texts) == 0:
    print("ERROR: No text chunks found in JSON file.")
    exit()

print("Generating embeddings (this may take a moment)...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("Creating FAISS index...")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Total vectors stored in FAISS: {index.ntotal}")

# Save FAISS index
faiss.write_index(index, "brain_index.faiss")

# Save corresponding texts
with open("texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f)

print("FAISS index saved as brain_index.faiss")
print("Text mapping saved as texts.json")
print("Embedding process completed successfully.")
