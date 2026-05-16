# SETUP.md - Detailed Installation & Configuration Guide

## System Requirements

- **Python Version:** 3.8 or higher
- **RAM:** Minimum 4GB (8GB recommended for large documents)
- **Storage:** ~2GB for dependencies and models
- **Operating System:** Windows, macOS, or Linux

## Step-by-Step Installation

### 1. Prepare Your Environment

**On Windows (PowerShell):**
```powershell
# Navigate to project directory
cd C:\Users\YourUsername\Documents\rag-data-assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\Activate.ps1
# If you get execution policy error, use:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On macOS/Linux:**
```bash
# Navigate to project directory
cd ~/Documents/rag-data-assistant

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Expected Installation Time:** 3-5 minutes
**Download Size:** ~1.5GB (includes PyTorch and models)

### 3. Verify Installation

```bash
python -c "import streamlit; import torch; import faiss; print('✅ All packages installed successfully!')"
```

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Stop the Application
- Press `Ctrl+C` in the terminal
- Or close the browser tab and wait 30 seconds

## First-Time Usage

### Demo Workflow

1. **Open the App**
   - Browser will open at `http://localhost:8501`
   - If not, manually navigate to that URL

2. **Upload Sample Document**
   - Click "Choose a text file to upload"
   - Select `sample_document.txt` from the project folder
   - Wait for "✅ Document processed!" message

3. **Explore Configuration**
   - In the left sidebar, adjust "Chunk Size" (try 300-500)
   - Adjust "Chunk Overlap" (try 50)
   - These settings affect retrieval quality

4. **Ask Questions**
   - Try: "What is RAG?"
   - Try: "What are the benefits?"
   - Try: "Tell me about technical implementation"

5. **Observe Results**
   - View retrieved chunks with relevance scores
   - See generated answer based on context
   - Experiment with different questions

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'streamlit'`

**Solution:** Ensure virtual environment is activated
```bash
# Check if activated (should show "venv" in terminal)
# If not, activate it:
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate    # macOS/Linux

# Reinstall
pip install -r requirements.txt
```

### Issue: `torch.cuda.OutOfMemoryError` or slow performance

**Solution:** Use CPU-only FAISS (already configured)
- If needed, reduce chunk size to 300
- Close other applications
- Process smaller documents first

### Issue: First run takes a very long time

**Solution:** This is normal!
- Streamlit caches the Sentence-Transformer model
- First run downloads ~133 MB model
- Subsequent runs start in <2 seconds
- The model is cached in `~/.cache/huggingface/`

### Issue: "No module named utils"

**Solution:** Ensure you're running from project root
```bash
# Wrong location
cd Desktop
streamlit run app.py  # ❌ Can't find utils

# Correct location
cd rag-data-assistant
streamlit run app.py  # ✅ Works
```

### Issue: Port 8501 already in use

**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Document upload doesn't work

**Solution:** 
- Ensure file is `.txt` format
- Check file encoding is UTF-8
- Try a smaller test file first

### Issue: No chunks created

**Possible causes:**
- Document might be too short (< chunk size)
- Try reducing chunk size in sidebar
- Try uploading a different document

### Issue: Slow query response

**Solution:**
- Close other applications to free memory
- Reduce number of chunks (upload shorter document)
- Reduce chunk size
- Pre-process document before uploading

## Configuration Customization

### Change Embedding Model

Edit `utils/retriever.py`, line 15:

```python
# Current (fast, good quality)
self.model = SentenceTransformer("all-MiniLM-L6-v2")

# Alternative options:
# self.model = SentenceTransformer("all-mpnet-base-v2")  # Better quality, slower
# self.model = SentenceTransformer("all-DistilBERT-v1")  # Faster, lower quality
```

### Adjust Default Chunk Size

Edit `app.py`, around line 45:

```python
chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 1000, 400, step=50)  # Change 400
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", 0, 100, 50, step=10)
```

### Customize UI Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"       # Change these hex colors
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## Performance Optimization

### For Faster Startup
```bash
# Clear Streamlit cache
streamlit cache clear
```

### For Large Documents
1. Split document into smaller parts
2. Process separately
3. Or reduce chunk size

### For Better Retrieval
1. Increase chunk size to 600-800
2. Use more relevant documents
3. Ask more specific questions

## Production Deployment

### Local Network Access
```bash
streamlit run app.py --server.address 0.0.0.0
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Run:
```bash
docker build -t rag-assistant .
docker run -p 8501:8501 rag-assistant
```

### Streamlit Cloud Deployment
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Deploy with one click!

## Important Notes

- ⚠️ **First embedding generation takes time** - Model is downloaded and cached
- 💾 **Models are cached locally** - Subsequent runs are fast
- 🔐 **No data is sent to external servers** - Everything runs locally
- 📊 **Memory usage scales with document size** - Larger docs = more RAM needed
- 🔄 **Index rebuilds on each document upload** - This is expected behavior

## Getting Help

1. Check the [README.md](README.md) for detailed documentation
2. Review [QUICKSTART.md](QUICKSTART.md) for basic usage
3. Check error messages - they usually point to the solution
4. Ensure virtual environment is activated before running
5. Try clearing Streamlit cache: `streamlit cache clear`

## Next Steps

After successful setup:
1. Try different documents
2. Experiment with different chunk sizes
3. Add custom documents
4. Customize answer generation (see `utils/generator.py`)
5. Consider local LLM integration with Ollama
6. Deploy to Streamlit Cloud for sharing

---

**Happy exploring!** 🚀
