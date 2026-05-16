# Quick Start Guide for RAG Data Assistant

## Installation

```bash
cd rag-data-assistant
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## First Use

1. Upload `sample_document.txt` or a file from `sample_documents/`.
2. View document statistics after processing.
3. Try asking: "What is RAG?"
4. Try asking: "What are the benefits of RAG?"
5. View the retrieved chunks and generated answer in separate sections.

## Key Features

✅ Upload `.txt`, `.pdf`, and `.csv` documents
✅ Automatic document parsing and sentence-aware chunking
✅ Configurable chunk size and chunk overlap
✅ Local embeddings with Sentence-Transformers
✅ Fast exact vector search with FAISS `IndexFlatL2`
✅ Q&A with top 3 retrieved chunks
✅ Retrieved context shown separately from the generated answer
✅ Clean Streamlit UI

## Project Structure

- `app.py` - Main Streamlit interface
- `utils/document_loader.py` - TXT/PDF/CSV loading and chunking
- `utils/retriever.py` - Embedding and FAISS search
- `utils/generator.py` - Extractive answer synthesis
- `tests/` - Lightweight unit tests
- `sample_document.txt` - Main demo document
- `sample_documents/` - Additional CSV/PDF demo assets
- `requirements.txt` - Python dependencies

## Customization Ideas

- Add DOCX support.
- Add OCR for scanned PDFs.
- Integrate a local LLM with Ollama for stronger answer generation.
- Add conversation history.
- Persist indexed documents between app restarts.

See `README.md` for full documentation.
