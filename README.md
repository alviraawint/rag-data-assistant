# RAG Data Assistant: Local Document Search & Retrieval-Grounded Q&A

> A beginner-friendly but professional Retrieval-Augmented Generation (RAG)-style Streamlit app for uploading `.txt`, `.pdf`, and `.csv` files, converting document text into embeddings, retrieving relevant chunks with FAISS, and producing concise answers from retrieved context.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io)
[![No API Key Required](https://img.shields.io/badge/No%20API%20Key-Required-green.svg)](#current-scope)

---

## Business Problem

Organizations store useful knowledge in PDFs, CSV exports, internal notes, reports, manuals, and support documents. Traditional keyword search can miss relevant information when users ask questions in different words, while general-purpose LLMs can hallucinate if they are not grounded in source material.

This project demonstrates how a lightweight RAG pipeline can help users ask natural-language questions over uploaded documents and inspect the retrieved source context behind each answer.

Example internship-relevant use cases:

- **Customer support:** search help-center articles and troubleshooting notes.
- **Operations and logistics:** query process manuals, shipment exception guides, or SOPs.
- **Healthcare and research:** search readable reports, study summaries, or internal documentation.
- **Finance and compliance:** inspect policy documents or CSV exports with transparent evidence.
- **Engineering and field service:** search product notes, service manuals, and knowledge-base articles.

---

## Current Scope

This project is intentionally local-first and does not require paid API keys.

Current answer generation is **extractive**: the app selects relevant sentences from the retrieved chunks and formats them into a concise answer. This keeps the demo lightweight, transparent, and easy for recruiters to run, but it is not the same as using a large language model for abstractive generation.

Planned extension: add an optional LLM generation backend using Ollama, OpenAI, or Hugging Face while preserving retrieved source context.

---

## Core Features

- Upload and parse `.txt`, `.pdf`, and `.csv` files.
- Clean and split documents into sentence-aware chunks with configurable overlap.
- Encode chunks with Sentence-Transformers embeddings.
- Store vectors in a FAISS `IndexFlatL2` index for exact vector similarity search.
- Embed user questions and retrieve the top matching chunks.
- Show retrieved context with relevance scores separately from the final answer.
- Run locally with Streamlit and no API keys.

---

## How the RAG Pipeline Works

| Step | Description | Implementation |
|---|---|---|
| 1. Upload | User uploads a `.txt`, `.pdf`, or `.csv` file | `app.py` |
| 2. Parse | Extract readable text from the file | `utils/document_loader.py` |
| 3. Chunk | Split cleaned text into smaller overlapping passages | `utils/document_loader.py` |
| 4. Embed | Convert chunks into dense vectors | `utils/retriever.py` |
| 5. Index | Store vectors in a FAISS `IndexFlatL2` index | `utils/retriever.py` |
| 6. Retrieve | Embed the query and return top-k similar chunks | `utils/retriever.py` |
| 7. Answer | Select relevant sentences from retrieved context | `utils/generator.py` |
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
| Vector search | FAISS `IndexFlatL2` | Exact nearest-neighbor search for demo-scale documents |
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

## Recruiter Demo in 60 Seconds

1. Install dependencies with `pip install -r requirements.txt`.
2. Start the app with `streamlit run app.py`.
3. Upload `sample_document.txt` or a file from `sample_documents/`.
4. Ask: **“What is RAG?”**
5. Ask: **“How does RAG reduce hallucination?”**
6. Ask: **“What are the main technical components?”**
7. Compare the retrieved context with the generated answer.

What to notice:

- The app chunks the document automatically.
- Sentence-Transformers converts chunks and questions into embeddings.
- FAISS retrieves the top matching chunks.
- Retrieved context is shown separately from the final answer for transparency.

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
│   ├── retriever.py               # Sentence-Transformers + FAISS retrieval
│   ├── generator.py               # Extractive answer synthesis and context display
│   └── __init__.py
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

## Evaluation Ideas

This project can be evaluated with a small set of document-question pairs:

| Question | Expected Evidence |
|---|---|
| What is RAG? | Definition from the introduction section of `sample_document.txt`. |
| What are the benefits of RAG? | Benefits section mentioning current information, reduced hallucination, and transparency. |
| What technologies are used in modern RAG systems? | Technical implementation section mentioning embeddings and vector databases. |
| Which region has the highest delivery volume? | CSV sample rows in `sample_documents/sample_business_metrics.csv`. |

Potential future metrics:

- Retrieval precision@k.
- Mean reciprocal rank.
- Answer faithfulness.
- Query latency.
- User feedback score.

---

## Limitations

- Current answers are extractive, not LLM-generated.
- The FAISS index is stored in memory and rebuilt after app restart.
- The app supports one uploaded document at a time.
- Scanned PDFs and image-only documents are not supported.
- The current FAISS index is exact `IndexFlatL2`, not a production-scale approximate index such as IVF or HNSW.
- The app is intended for small demo documents, not enterprise-scale document collections.

---

## Future Improvements

### Retrieval Quality

- Add hybrid retrieval with BM25 plus embeddings.
- Add a cross-encoder reranker for the top retrieved chunks.
- Add source metadata such as page number, row number, and chunk ID.

### Generation Quality

- Add an optional local LLM backend with Ollama.
- Add prompt templates with explicit grounding instructions.
- Add answer citations that point to specific retrieved chunks.

### Productization

- Add multi-document upload and persistent vector indexes.
- Add Docker support and Streamlit Cloud deployment instructions.
- Add a feedback button for answer quality.
- Add OCR for scanned PDFs.

---

## Skills Demonstrated

### AI / Machine Learning

- Dense embeddings with Sentence-Transformers.
- Semantic search and vector similarity.
- Retrieval pipeline design.
- RAG concepts: chunking, retrieval, grounding, and transparency.

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

No API key is required. The first run may create a local `.cache/` directory for the Hugging Face embedding model. That directory, virtual environments, Streamlit secrets, local FAISS indexes, and `.env` files are excluded by `.gitignore` and should not be committed.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
