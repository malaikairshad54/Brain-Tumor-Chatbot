from flask import Flask, render_template, request, jsonify
import json
import faiss
import numpy as np
import requests
import os
import uuid
import datetime
import warnings

# Suppress benign warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ----------------------------
# CONFIG & SETUP
# ----------------------------

CHATS_DIR = "chats"
if not os.path.exists(CHATS_DIR):
    os.makedirs(CHATS_DIR)

# ----------------------------
# LOAD MODELS & DATA (RUNS ONCE)
# ----------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("brain_index.faiss")

with open("texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# ----------------------------
# Retrieval / QA config
# ----------------------------
QUESTIONS_PATH = os.path.join("data", "questions_answers.json")
RETRIEVAL_DISTANCE_THRESHOLD = 0.8  # adjust as needed (L2 distance)
MAX_CONTEXT_CHARS = 2000
KEYWORD_VERIFICATION = True

# Disclaimer appended to every bot answer
DISCLAIMER_HTML = """
<div class="disclaimer">
    <strong>Disclaimer:</strong> Remember, this information is for educational purposes only and does not substitute medical diagnosis or treatment. If you have any concerns about brain tumors, please seek the guidance of a qualified healthcare provider.
</div>
"""

# Load Q&A file for pattern matching
qa_data = []
if os.path.exists(QUESTIONS_PATH):
    try:
        with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
    except Exception:
        qa_data = []

print("System ready.")

# ----------------------------
# MEMORY & FILE OPERATIONS
# ----------------------------

def get_chat_filepath(chat_id):
    return os.path.join(CHATS_DIR, f"{chat_id}.json")

def load_chat(chat_id):
    filepath = get_chat_filepath(chat_id)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"id": chat_id, "title": "New Chat", "messages": [], "timestamp": str(datetime.datetime.now())}

def save_chat(chat_id, chat_data):
    filepath = get_chat_filepath(chat_id)
    chat_data["timestamp"] = str(datetime.datetime.now()) # Update timestamp on save
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=4)

def list_chats():
    chats = []
    if not os.path.exists(CHATS_DIR):
        return []
        
    for filename in os.listdir(CHATS_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHATS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    chats.append({
                        "id": data.get("id", filename.replace(".json", "")),
                        "title": data.get("title", "Untitled Chat"),
                        "timestamp": data.get("timestamp", "")
                    })
            except:
                continue
    
    # Sort by timestamp descending
    chats.sort(key=lambda x: x["timestamp"], reverse=True)
    return chats

# ----------------------------
# SEARCH FUNCTION
# ----------------------------

def search(query, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    return [texts[idx] for idx in indices[0]]


def retrieve_documents(query, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({"text": texts[idx], "distance": float(dist), "index": int(idx)})
    return results


def normalize_text(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()


def match_qa_pattern(question: str):
    q_norm = normalize_text(question)
    for item in qa_data:
        patterns = item.get("patterns", [])
        for p in patterns:
            p_norm = normalize_text(p)
            if p_norm == q_norm or p_norm in q_norm or q_norm in p_norm:
                return item.get("response")
    return None


def question_keywords(question: str):
    stopwords = set(["the","and","is","in","of","a","an","to","for","with","on","are","how","what","why","when","which","do","does"]) 
    words = [w for w in normalize_text(question).split() if len(w) > 2 and w not in stopwords]
    return set(words)


def keywords_verify(question: str, docs: list, min_overlap: int = 1) -> bool:
    if not KEYWORD_VERIFICATION:
        return True
    keys = question_keywords(question)
    if not keys:
        return True
    combined = " ".join(d.get("text", "") for d in docs).lower()
    overlap = sum(1 for k in keys if k in combined)
    return overlap >= min_overlap

# ----------------------------
# LLM CALL FUNCTION
# ----------------------------

def ask_mistral(history, documents, question):
    # Enforce strict rules: answers must be rewritten only from retrieved documents.
    prompt = f"""
You are a controlled medical information formatter.

IMPORTANT SYSTEM RULES — YOU MUST FOLLOW ALL:

1. You are NOT allowed to use your pretrained knowledge.
2. You are NOT allowed to add external medical information.
3. You are NOT allowed to fill missing gaps.
4. You must ONLY use the provided CONTEXT section.
5. If the answer is not explicitly stated in the CONTEXT, respond EXACTLY with:

"I cannot find this information in the provided medical documents."

6. If the question is unrelated to brain tumors, respond EXACTLY with:

"This assistant only provides information about brain tumors."

7. Do NOT generate assumptions.
8. Do NOT generate explanations beyond the provided text.
9. Do NOT invent examples.
10. Do NOT provide reasoning steps.
11. Maximum answer length: 120 words.

YOUR ROLE:
You are NOT the knowledge source.
You are ONLY responsible for:

• Rewriting the retrieved medical content clearly
• Structuring it properly
• Formatting into bullet points when needed
• Keeping language simple and professional

FORMAT RULES:

- Definitions → 2–3 short sentences.
- Symptoms / Treatments / Types → bullet points only.
- No introduction sentence.
- No concluding remarks.
- No external commentary.
- No repetition of the question.
- No creative writing.

CONTEXT:
----------------
{documents}
----------------

QUESTION:
{question}

Now produce the final answer strictly using ONLY the context above.
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "stop": ["Question:", "User:", "Bot:"] # Stop generation if it tries to continue the chat pattern
                }
            }
        )
        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            # Safety cleanup if model still chats
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            # Remove artifacts commonly generated by Phi/base models
            artifacts = ["SOLVED CHALLENGE:", "Assistant:", "ANSWER:", "Solution:"]
            for artifact in artifacts:
                answer = answer.replace(artifact, "")
            
            answer = answer.strip()
            return answer
        else:
            return "Error: LLM service not available."
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chats", methods=["GET"])
def get_chats():
    return jsonify(list_chats())

@app.route("/chat/<chat_id>", methods=["GET"])
def get_chat_history(chat_id):
    return jsonify(load_chat(chat_id))

@app.route("/new_chat", methods=["POST"])
def create_new_chat():
    chat_id = str(uuid.uuid4())
    new_chat = {
        "id": chat_id,
        "title": "New Chat",
        "messages": [],
        "timestamp": str(datetime.datetime.now())
    }
    save_chat(chat_id, new_chat)
    return jsonify(new_chat)

@app.route("/delete_chat/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    filepath = get_chat_filepath(chat_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found"}), 404

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")
    chat_id = data.get("chat_id")

    if not chat_id:
        chat_id = str(uuid.uuid4()) # Fallback if no chat_id provided

    # Load current chat history
    chat_data = load_chat(chat_id)
    
    # Update title if it's the first message
    if len(chat_data["messages"]) == 0:
        chat_data["title"] = user_question[:30] + "..." if len(user_question) > 30 else user_question

    # Retrieve relevant chunks
    # 1) Check Q&A pattern match first — return stored answer without LLM if matched
    matched_answer = match_qa_pattern(user_question)
    if matched_answer:
        final_answer = matched_answer + DISCLAIMER_HTML
        # Append to chat history and save
        chat_data["messages"].append({"sender": "user", "text": user_question, "timestamp": str(datetime.datetime.now())})
        chat_data["messages"].append({"sender": "bot", "text": final_answer, "timestamp": str(datetime.datetime.now())})
        save_chat(chat_id, chat_data)
        return jsonify({"answer": final_answer, "chat_id": chat_id, "title": chat_data["title"]})

    # 2) FAISS retrieval
    retrieved = retrieve_documents(user_question, top_k=5)
    if not retrieved:
        return jsonify({"answer": "I cannot find this information in the provided medical documents.", "chat_id": chat_id, "title": chat_data["title"]})

    # If best distance exceeds threshold -> consider not found
    best = retrieved[0]
    if best.get("distance", 1e9) > RETRIEVAL_DISTANCE_THRESHOLD:
        return jsonify({"answer": "I cannot find this information in the provided medical documents.", "chat_id": chat_id, "title": chat_data["title"]})

    # Keyword verification (optional)
    if not keywords_verify(user_question, retrieved):
        return jsonify({"answer": "I cannot find this information in the provided medical documents.", "chat_id": chat_id, "title": chat_data["title"]})

    # Build combined_context with size control
    combined_context = ""
    for doc in retrieved:
        if len(combined_context) + len(doc["text"]) + 2 > MAX_CONTEXT_CHARS:
            break
        if combined_context:
            combined_context += "\n\n"
        combined_context += doc["text"]

    # Add memory (last 3 exchanges from this specific chat)
    memory_context = ""
    # Filter for valid Q&A pairs (user/bot)
    history_pairs = []
    temp_q = None
    for msg in chat_data["messages"]:
        if msg["sender"] == "user":
            temp_q = msg["text"]
        elif msg["sender"] == "bot" and temp_q:
            history_pairs.append({"question": temp_q, "answer": msg["text"]})
            temp_q = None

    for entry in history_pairs[-3:]:
        memory_context += f"User: {entry['question']}\nBot: {entry['answer']}\n"

    # Call LLM with separated contexts
    final_answer = ask_mistral(memory_context, combined_context, user_question)
    # Append disclaimer to LLM answer
    final_answer = final_answer + DISCLAIMER_HTML

    # Append to chat history
    chat_data["messages"].append({"sender": "user", "text": user_question, "timestamp": str(datetime.datetime.now())})
    chat_data["messages"].append({"sender": "bot", "text": final_answer, "timestamp": str(datetime.datetime.now())})
    
    save_chat(chat_id, chat_data)

    return jsonify({"answer": final_answer, "chat_id": chat_id, "title": chat_data["title"]})


if __name__ == "__main__":
    app.run(debug=True)
