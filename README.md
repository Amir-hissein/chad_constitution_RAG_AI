# 🇹🇩 Constitution of Chad — RAG Assistant

> A Retrieval-Augmented Generation (RAG) assistant that answers questions about the
> **Constitution of Chad (2023, revised 2025)** using only the official text — with
> article-level citations and no hallucinations.


**Stack:** Python · LangChain · ChromaDB · BGE-M3 (embeddings) · Ollama (Llama 3.1) · LangGraph

---



### Overview

Large Language Models do not know the Constitution of Chad, and if asked directly they
tend to **invent** answers (hallucinate). This project solves that with **RAG**: before
answering, the system **retrieves the most relevant articles** from the real document and
feeds them to the LLM with a strict instruction — *"answer only from this context and cite
the article numbers."*

The result is a trustworthy legal assistant that responds in French (the official language
of the text), grounds every answer in real articles, and refuses when the information is not
in the Constitution.

### How RAG Works (Architecture)

The pipeline is split into two phases:

```
PHASE A — INDEXING (run once, offline)
  1. Load      load_pdf.py          PDF  → clean raw text
  2. Split     split_documents.py   text → chunks (one per article + metadata)
  3. Embed     build_index.py       chunk → vector (BGE-M3)
  4. Index     build_index.py       vectors → ChromaDB (persisted on disk)

PHASE B — QUERYING (run on every question)
  5. Retrieve  rag_chain.py         question → 6 closest articles (semantic search)
  6. Generate  rag_chain.py         articles + question → grounded answer (Llama 3.1)
  7. Orchestrate  graph.py          retrieve → generate as a LangGraph state machine

INTERFACE & EVALUATION
  8. Web UI    app.py               ask questions in the browser (Streamlit)
  9. Evaluate  evaluate.py          score retrieval & answers on eval/questions.json
```

**Key idea — embeddings.** An embedding turns a text into a list of ~1024 numbers (a
*vector*) such that texts with similar *meaning* have close vectors. Searching for *"voting
age"* therefore matches the article that says *"âgés de dix-huit ans"* (eighteen years old)
even though the word "age" never appears — semantic search, not keyword search.

**Golden rule.** The *same* embedding model (BGE-M3) is used both to index and to query.
Otherwise the vectors are not comparable.

### Project Structure

```
chad_constitution_RAG/
├── data/
│   └── constitution_tchad.pdf     # source document (text PDF, not scanned)
├── src/
│   ├── load_pdf.py                # Step 1 — load & clean the PDF
│   ├── split_documents.py         # Step 2 — chunk by article + metadata
│   ├── build_index.py             # Steps 3-4 — embeddings + ChromaDB index
│   ├── rag_chain.py               # Steps 5-6 — retrieval + generation
│   ├── graph.py                   # Step 7 — LangGraph orchestration (optional)
│   ├── app.py                     # Step 8 — Streamlit web interface
│   └── evaluate.py                # Step 9 — automatic evaluation
├── eval/
│   └── questions.json             # Q&A test set used by evaluate.py
├── chroma_db/                     # persisted vector store (auto-generated)
├── requirements.txt
├── .vscode/settings.json          # points the editor to the venv
└── README.md
```

### Requirements

- **Python 3.12+**
- **Ollama** (local, free LLM runtime) — https://ollama.com
- ~7 GB free disk (BGE-M3 model ~2.3 GB + Llama 3.1 ~4.9 GB)

### Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\Activate.ps1          # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the local LLM
ollama pull llama3.1
```

### Usage

```bash
# Build the vector index (run once — downloads BGE-M3 on first run, a few minutes)
python src/build_index.py

# Option A — Web interface (recommended): opens http://localhost:8501 in your browser
streamlit run src/app.py

# Option B — Command line
python src/rag_chain.py

# Optional — measure quality (retrieval & answers) on a fixed question set
PYTHONPATH=src python src/evaluate.py
```

The **web interface** (`app.py`) shows the generated answer plus an expandable panel listing
the exact articles that were retrieved, so you can check where each answer comes from. The
heavy model/index load is cached, so only the first question is slow.

Each step file is runnable on its own and prints a ✅ / ❌ self-test, so you can validate the
pipeline stage by stage.

### Example

```
Question: Quelles sont les conditions pour être candidat à la présidence ?

Retrieved: Article 68, Article 69, Article 72
Answer:    Pour être candidat à l'élection présidentielle, il faut … (Article 68).
```

If a question is out of scope (e.g. *"What is the capital of France?"*), the assistant
replies: *"Cette information ne figure pas dans la Constitution."*

### Design Notes

- **Chunking by article** keeps each legal unit intact instead of cutting blindly every N
  characters. Very long articles (> 1500 chars) are further split with a
  `RecursiveCharacterTextSplitter`.
- **Metadata** (`article`, `titre`, `source`) travels with each chunk all the way to the
  answer, enabling precise citations.
- **The LLM is a swappable module.** Retrieval, embeddings, and the vector store are
  independent of the generator — you can replace Llama 3.1 with any cloud model (Claude,
  GPT-4o, Gemini) by changing a single line in `rag_chain.py`.

### Known Limitations

- The document preamble (text before Article 1) is intentionally dropped by the article
  splitter.
- On the built-in evaluation set, retrieval is ~80%: a few "opening" articles of a title
  (e.g. 65, 110) are occasionally missed. Raising `k` in `rag_chain.py` helps.

---

