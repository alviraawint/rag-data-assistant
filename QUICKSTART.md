# Quick Start Guide for RAG Data Assistant

## Installation
```bash
cd rag-data-assistant
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app.py
```

The app will open at http://localhost:8501

## First Use
1. Click "Choose a text file" and upload `sample_document.txt`
2. View the document statistics
3. Try asking: "What is RAG?" or "What are the benefits of RAG?"
4. View retrieved chunks and generated answer

## Project Structure
- `app.py` - Main Streamlit interface
- `utils/document_loader.py` - Document chunking
- `utils/retriever.py` - Embedding & search with FAISS
- `utils/generator.py` - Answer generation
- `requirements.txt` - Dependencies
- `sample_document.txt` - Example document

## Key Features
✅ Upload .txt documents
✅ Automatic intelligent chunking
✅ AI-powered embeddings (free, local)
✅ Fast vector search with FAISS
✅ Q&A with top 3 relevant chunks
✅ Clean, responsive UI

## Customization Ideas
- Add PDF/DOCX support
- Integrate local LLM (Ollama) for better answers
- Add conversation history
- Persist indexed documents

See README.md for full documentation!
