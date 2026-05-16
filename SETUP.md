# Detailed Setup Guide

This guide helps you run the RAG Data Assistant locally.

## 1. Create a Virtual Environment

```bash
cd rag-data-assistant
python -m venv venv
```

Activate it:

```bash
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows PowerShell
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 is recommended. The first run may download the Sentence-Transformers embedding model into a local `.cache/` directory. This cache is ignored by Git.

## 3. Run the Application

```bash
streamlit run app.py
```

Expected local URL:

```text
http://localhost:8501
```

## 4. Demo Workflow

1. Upload `sample_document.txt`, `sample_documents/sample_business_metrics.csv`, or `sample_documents/sample_policy_document.pdf`.
2. Adjust chunk size and chunk overlap in the sidebar if desired.
3. Ask a natural-language question.
4. Review the retrieved context and generated answer separately.

Good starter questions:

- What is RAG?
- What are the benefits of RAG?
- What are the main technical components?
- Which region has the highest delivery volume?
- What does the sample policy say about reviewing source context?

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit'`

Activate your virtual environment and reinstall dependencies:

```bash
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

### First run is slow

This is expected. The embedding model is downloaded the first time and cached locally for later runs.

### PDF upload has no readable text

The app supports text-based PDFs. Scanned/image-only PDFs need OCR, which is not implemented yet.

### CSV output looks verbose

The CSV loader converts each row into text with column names so semantic search can retrieve row-level evidence.

### Port 8501 is already in use

Use a different port:

```bash
streamlit run app.py --server.port 8502
```

## Optional Checks

Run the lightweight unit tests:

```bash
python -m unittest discover -s tests
```

Compile Python files to catch syntax errors:

```bash
python -m compileall app.py utils tests
```
