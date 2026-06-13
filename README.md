# RAG Data Assistant: Local Document Search & Retrieval-Grounded Q&A

> A beginner-friendly but professional Retrieval-Augmented Generation (RAG)-style Streamlit app for uploading `.txt`, `.pdf`, and `.csv` files, converting document text into embeddings, retrieving relevant chunks with FAISS, and producing concise answers from retrieved context.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io)
[![Claude Generation: Optional](https://img.shields.io/badge/Claude%20Generation-Optional-8A2BE2.svg)](#current-scope)

---

## Business Problem

Organizations store useful knowledge in PDFs, CSV exports, internal notes, reports, manuals, and support documents. Traditional keyword search can miss relevant information when users ask questions in different words, while general-purpose LLMs can hallucinate if they are not grounded in source material.

This project demonstrates how a lightweight RAG pipeline can help users ask natural-language questions over uploaded documents and inspect the retrieved source context behind each answer.

Relevant use cases:

- **Customer support:** search help-center articles and troubleshooting notes.
- **Operations and logistics:** query process manuals, shipment exception guides, or SOPs.
- **Healthcare and research:** search readable reports, study summaries, or internal documentation.
- **Finance and compliance:** inspect policy documents or CSV exports with transparent evidence.
- **Engineering and field service:** search product notes, service manuals, and knowledge-base articles.

---

## Current Scope

This project is local-first and runs without any paid API key. Answer generation has two modes:

- **Claude-grounded (recommended):** when an `ANTHROPIC_API_KEY` is set, retrieved chunks are sent to Anthropic Claude with a strict "answer only from the context" prompt. Answers are streamed and cite their sources inline as `[Chunk N]`.
- **Extractive fallback (no key):** when no key is configured, the app selects the most relevant sentences from the retrieved chunks and formats them into a concise answer — lightweight, transparent, and zero-dependency.

The retrieved context is always shown alongside the answer so it can be verified against the source.

---

## Core Features

- Upload and parse `.txt`, `.pdf`, and `.csv` files.
- Clean and split documents into sentence-aware chunks with configurable overlap.
- Encode chunks with Sentence-Transformers embeddings.
- Store vectors in a FAISS `IndexFlatIP` index for exact cosine-similarity search.
- Embed user questions and retrieve the top matching chunks.
- Generate grounded, cited answers with Claude, or fall back to an extractive answer with no API key.
- Show retrieved context with relevance scores separately from the final answer.
- Measure retrieval quality (Hit@k, Recall@k, MRR) with the included evaluation harness.

---

## How the RAG Pipeline Works

| Step | Description | Implementation |
|---|---|---|
| 1. Upload | User uploads a `.txt`, `.pdf`, or `.csv` file | `app.py` |
| 2. Parse | Extract readable text from the file | `utils/document_loader.py` |
| 3. Chunk | Split cleaned text into smaller overlapping passages | `utils/document_loader.py` |
| 4. Embed | Convert chunks into dense vectors | `utils/retriever.py` |
| 5. Index | Store vectors in a FAISS `IndexFlatIP` (cosine) index | `utils/retriever.py` |
| 6. Retrieve | Embed the query and return top-k similar chunks | `utils/retriever.py` |
| 7. Answer | Generate a grounded, cited answer with Claude (or extractive fallback) | `utils/llm_generator.py`, `utils/generator.py` |
| 8. Display | Show retrieved context and answer in separate sections | `app.py` |

### RAG Concept in Plain English

1. **Document upload:** the user provides a document instead of relying on a model's built-in memory.
2. **Chunking:** the document is split into smaller passages so retrieval can target the most relevant parts.
3. **Embeddings:** each chunk becomes a dense numeric vector that captures semantic meaning.
4. **Vector search:** FAISS compares the question vector against chunk vectors.
5. **Retrieval:** the app returns the top matching chunks with relevance scores.
6. **Answer synthesis:** the app creates a concise answer using only the retrieved context.
7. **Transparency:** the retrieved context remains visible so users can verify the answer.

---

## Supported File Types

| File Type | Status | Notes |
|---|---|---|
| `.txt` | Supported | Reads UTF-8 text files. |
| `.pdf` | Supported | Extracts readable text from PDF pages with `pypdf`; scanned or image-only PDFs need OCR and are not currently supported. |
| `.csv` | Supported | Converts rows into readable text using column names and row numbers. |
| `.docx` | Not yet supported | Good future extension. |
| Images / scanned PDFs | Not yet supported | Would require OCR. |

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| App framework | Streamlit | Interactive web UI |
| Embeddings | Sentence-Transformers | Converts text into semantic vectors |
| Generation | Anthropic Claude (`claude-opus-4-8`) | Grounded answer generation over retrieved context (optional; extractive fallback) |
| Vector search | FAISS `IndexFlatIP` | Exact cosine-similarity search for demo-scale documents |
| Data processing | Pandas | CSV loading and row-to-text conversion |
| PDF parsing | pypdf | Extracts text from readable PDF pages |
| Numerical computing | NumPy | Embedding array handling |
| ML backend | PyTorch | Backend used by Sentence-Transformers |

**Tested Python version:** Python 3.10 is recommended for the smoothest setup. The project targets Python 3.8+.

---

## Screenshots & Usage Examples

### Home Page & Document Upload

The main screen keeps the workflow simple: upload a TXT, PDF, or CSV file, adjust chunking settings, and view document statistics after processing.

![RAG Data Assistant home page with upload and document statistics](screenshots/Home%20page.png)

### Asking a Question

After a document is indexed, users can ask a natural language question. The app keeps the retrieved chunks visible so the answer can be checked against the source text.

![Question input and retrieved context](screenshots/Ask%20a%20question.png)

### Generated Answer

The generated answer is shown separately from the retrieved context, giving users a concise response while preserving transparency.

![Generated answer section](screenshots/Generated%20answer.png)

---


## How to Run Locally

### Prerequisites

- Python 3.8+; Python 3.10 recommended.
- 4 GB RAM minimum; 8 GB recommended.
- Internet connection on first run to download the embedding model.

### Installation

```bash
cd rag-data-assistant
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

### Enable Claude answers (optional)

```bash
cp .env.example .env   # then set ANTHROPIC_API_KEY in .env
```

Without a key the app uses the extractive fallback. With a key, answers are generated by Claude and grounded in the retrieved context. Override the model with `RAG_LLM_MODEL` (e.g. `claude-haiku-4-5`).

### Run the App

```bash
streamlit run app.py
```

Open the local Streamlit URL, usually `http://localhost:8501`.

---

## Project Structure

```text
rag-data-assistant/
├── app.py                         # Main Streamlit application
├── utils/
│   ├── document_loader.py         # TXT/PDF/CSV parsing and chunking
│   ├── retriever.py               # Sentence-Transformers + FAISS (cosine) retrieval
│   ├── llm_generator.py           # Claude-grounded answer generation
│   ├── generator.py               # Extractive answer fallback and context display
│   └── __init__.py
├── eval/                          # Retrieval + faithfulness evaluation harness
│   ├── evaluate.py
│   ├── qa_dataset.json
│   └── README.md
├── .env.example                   # Template for the optional ANTHROPIC_API_KEY
├── tests/
│   └── test_document_loader.py    # Lightweight unit tests for parsing/chunking
├── sample_documents/
│   ├── sample_business_metrics.csv
│   └── sample_policy_document.pdf
├── screenshots/                   # README screenshots
├── sample_document.txt            # Simple demo document
├── requirements.txt               # Python dependencies
├── QUICKSTART.md                  # Short setup guide
├── SETUP.md                       # Detailed setup notes
├── LICENSE                        # MIT license
└── README.md
```

---

## Evaluation

The pipeline is measured, not assumed. [`eval/evaluate.py`](eval/evaluate.py) runs a hand-labeled question set ([`eval/qa_dataset.json`](eval/qa_dataset.json)) through the retriever and reports standard information-retrieval metrics. With an API key it also runs an LLM-as-judge faithfulness check on the generated answers.

```bash
python eval/evaluate.py            # retrieval metrics (no API key needed)
python eval/evaluate.py --judge    # + answer faithfulness (needs ANTHROPIC_API_KEY)
```

**Results** (10 questions, `chunk_size=500`, exact cosine retrieval):

| Metric | k=3 | k=5 |
|---|---|---|
| Hit@k | 0.70 | 0.90 |
| Recall@k | 0.70 | 0.90 |
| MRR | 0.60 | 0.65 |

What the eval revealed (rather than assumed): most k=3 misses are *ranking* failures — the correct chunk is retrieved but ranked just outside the top-3 (hence the jump to 0.90 at k=5). That is the textbook motivation for the cross-encoder reranker listed under Future Improvements. See [`eval/README.md`](eval/README.md) for metric definitions and methodology.

---

## Limitations

- Without an API key, answers are extractive (not abstractive); Claude-grounded generation requires `ANTHROPIC_API_KEY`.
- The FAISS index is stored in memory and rebuilt after app restart.
- The app supports one uploaded document at a time.
- Scanned PDFs and image-only documents are not supported.
- The current FAISS index is exact `IndexFlatIP` (cosine), not a production-scale approximate index such as IVF or HNSW.
- The app is intended for small demo documents, not enterprise-scale document collections.

---

## Future Improvements

### Retrieval Quality

- Add hybrid retrieval with BM25 plus embeddings.
- Add a cross-encoder reranker for the top retrieved chunks.
- Add source metadata such as page number, row number, and chunk ID.

### Generation Quality

- Grounded prompt templates and inline `[Chunk N]` citations — **done** (Claude backend).
- Add an optional **local** LLM backend (e.g. Ollama) alongside the Claude backend.
- Add multi-turn / conversational follow-up questions.
- Promote the faithfulness eval into a CI regression gate.

### Productization

- Add multi-document upload and persistent vector indexes.
- Add Docker support and Streamlit Cloud deployment instructions.
- Add a feedback button for answer quality.
- Add OCR for scanned PDFs.

---

## Skills Demonstrated

### AI / Machine Learning

- Dense embeddings with Sentence-Transformers.
- Cosine-similarity semantic search with FAISS.
- Retrieval pipeline design.
- RAG concepts: chunking, retrieval, grounding, and transparency.
- LLM integration: grounded generation with Claude and inline citations.
- RAG evaluation: Hit@k, Recall@k, MRR, and LLM-as-judge faithfulness.

### Data Science / Data Engineering

- CSV parsing with Pandas.
- Text preprocessing and document conversion.
- Lightweight evaluation planning.

### Software Engineering

- Modular Python project structure.
- Streamlit UI development.
- Error handling for unsupported or unreadable files.
- GitHub-ready documentation, screenshots, samples, and tests.

---

## Notes on Local Files and Secrets

No API key is required to run the app (it falls back to extractive answers). To enable Claude-grounded answers, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The first run may create a local `.cache/` directory for the Hugging Face embedding model. That directory, virtual environments, Streamlit secrets, local FAISS indexes, generated eval results, and `.env` files are excluded by `.gitignore` and should not be committed.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
