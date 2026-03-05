# 🧠 Brain Tumor ChatBot

A **Retrieval-Augmented Generation (RAG)** powered chatbot specifically designed to answer questions about brain tumors. Built with **Flask**, **FAISS** vector search, **SentenceTransformers**, and a local **Ollama** LLM (Phi model), this system strictly answers from curated medical document chunks only — no hallucination, no external knowledge.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Disclaimer](#-disclaimer)
- [Contributing](#-contributing)

---

## ✨ Features

- 🔍 **Semantic Search** — Uses FAISS + SentenceTransformers (`all-MiniLM-L6-v2`) to retrieve the most relevant medical text chunks for any user query.
- 🤖 **Controlled LLM Answering** — A locally-hosted Phi model (via Ollama) rewrites retrieved content into clean, structured answers. It is strictly forbidden from using pretrained knowledge or hallucinating.
- 📂 **Pattern Matching Q&A** — Pre-curated Q&A pairs (`questions_answers.json`) are matched first for instant, consistent answers on common questions.
- 💬 **Multi-Session Chat History** — Each conversation is saved as a JSON file; users can create, load, and delete chat sessions.
- 🌐 **Flask Web Interface** — A clean, responsive web UI served via Flask for easy browser-based interaction.
- 🛡️ **Safety Guardrails** — Off-topic questions (not about brain tumors) are automatically deflected.
- 📝 **Medical Disclaimer** — Every answer is automatically appended with a medical disclaimer.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│              Flask Web App (app.py)           │
│                                              │
│  1. Pattern Match Q&A  ──► Instant Answer    │
│         │ (no match)                         │
│         ▼                                    │
│  2. FAISS Semantic Search                    │
│     (SentenceTransformer Embeddings)         │
│         │                                    │
│         ▼                                    │
│  3. Keyword Verification + Distance Check    │
│         │                                    │
│         ▼                                    │
│  4. Ollama (Phi model) — LLM Rewriter        │
│     [Context-only, strict prompt rules]      │
│         │                                    │
│         ▼                                    │
│  5. Final Answer + Disclaimer ──► User       │
└──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Brain Tumor bot/
│
├── app.py                   # Main Flask application & all routes
├── build_embeddings.py      # Script to build FAISS index from data
├── rag_chatbot.py           # Standalone CLI version of the RAG chatbot
├── check_index.py           # Utility to inspect the FAISS index
├── run_search.py            # Utility to test semantic search
│
├── data/
│   ├── chunks.json          # Medical document text chunks (source for embeddings)
│   └── questions_answers.json  # Curated Q&A pairs for pattern matching
│
├── templates/
│   └── index.html           # Frontend HTML/CSS/JS (Flask template)
│
├── static/
│   └── style.css            # Application stylesheet
│
├── chats/                   # Auto-generated: per-session chat JSON files (gitignored)
├── brain_index.faiss        # Auto-generated: FAISS vector index (gitignored)
├── texts.json               # Auto-generated: text mapping for FAISS (gitignored)
│
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## ✅ Prerequisites

Before you begin, ensure you have the following installed:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | [python.org](https://www.python.org/downloads/) |
| Ollama | Latest | [ollama.ai](https://ollama.ai) — for local LLM |
| Git | Latest | For cloning the repo |

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/malaikairshad54/Brain-Tumor-Chatbot.git
cd Brain-Tumor-Chatbot
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama & Pull the Phi Model

Make sure Ollama is running, then pull the model used by this app:

```bash
ollama pull phi
```

> **Note:** The app sends requests to `http://localhost:11434`. Ollama must be running in the background before launching the app.

### 5. Build the FAISS Embeddings Index

This step only needs to be done once (or whenever `data/chunks.json` or `data/questions_answers.json` is updated):

```bash
python build_embeddings.py
```

This will generate:
- `brain_index.faiss` — the vector search index
- `texts.json` — the text registry linked to the index

---

## ▶️ Running the Application

```bash
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 💡 How It Works

1. **User submits a question** via the web chat interface.
2. **Pattern Matching** — The question is normalized and compared against pre-defined Q&A patterns in `questions_answers.json`. If matched, a curated answer is returned immediately (no LLM call needed).
3. **FAISS Retrieval** — If no pattern match is found, the question is embedded using `all-MiniLM-L6-v2` and the top-5 closest document chunks are retrieved from the FAISS index.
4. **Quality Checks** — The best retrieval distance is checked against a threshold (`0.8`). A keyword overlap check also verifies relevance.
5. **LLM Rewriting** — If checks pass, the retrieved chunks + last 3 conversation turns are sent to the local Phi model via Ollama. The model is strictly instructed to only reformat the provided context.
6. **Response + Disclaimer** — The final answer is appended with a medical disclaimer and saved to the chat session.

---

## 📊 Dataset

The chatbot's knowledge comes from two data files:

| File | Description |
|---|---|
| `data/chunks.json` | Structured text chunks extracted from brain tumor medical documents |
| `data/questions_answers.json` | Curated Q&A pairs with patterns for fast, exact matching |

---

## ⚠️ Disclaimer

> This chatbot is for **educational purposes only**. It does **not** provide medical diagnoses or treatment advice. Always consult a qualified healthcare professional for any medical concerns related to brain tumors.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** this repository
2. **Create** a new branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "Add: your feature description"`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

Please ensure your code follows existing conventions and all edge cases are handled.

---


*Built with ❤️ for medical education and AI learning.*
