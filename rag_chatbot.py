import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load embedding model
# ----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Load FAISS index
# ----------------------------
print("Loading FAISS index...")
index = faiss.read_index("brain_index.faiss")

# ----------------------------
# Load stored texts
# ----------------------------
with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

print("System ready.\n")


# ----------------------------
# Search Function
# ----------------------------
def search(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(texts[idx])

    return results


# ----------------------------
# Ask Mistral (Improved Prompt)
# ----------------------------
def ask_mistral(context, question):

    prompt = f"""
A highly controlled medical information assistant specialized ONLY in brain tumors.

CRITICAL RULES:

1. You must answer ONLY using the provided context.
2. If the answer is not clearly found in the context, respond exactly with:
   "I cannot find this information in the provided medical documents."
3. If the question is unrelated to brain tumors, respond exactly with:
   "This assistant only provides information about brain tumors."
4. Do NOT generate creative explanations.
5. Do NOT add assumptions.
6. Do NOT provide external knowledge.
7. Keep the answer concise and structured.
8. Maximum length: 120 words.
9. Prefer bullet points for lists.
10. Do not repeat the question.
11. Do not include reasoning steps.
12. Do not hallucinate.

FORMAT RULES:

- For definitions → 2–3 short sentences.
- For symptoms/treatments/types → bullet points only.
- Keep sentences simple and direct.
- No extra commentary.

CONTEXT:
{context}

QUESTION:
{question}

At the end, ALWAYS add this disclaimer:
"This information is for educational purposes only and is not medical advice."
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3  # More factual, less random
            }
        }
    )

    return response.json()["response"]


# ----------------------------
# Main Loop
# ----------------------------
while True:
    user_question = input("\nAsk a question (or type 'exit'): ")

    if user_question.lower() == "exit":
        break

    print("\nSearching documents...")
    retrieved_chunks = search(user_question)

    combined_context = "\n\n".join(retrieved_chunks)

    print("Generating answer...")
    answer = ask_mistral(combined_context, user_question)

    print("\nAnswer:\n")
    print(answer)
