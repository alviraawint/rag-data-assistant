"""
Streamlit RAG Data Assistant - Main Application
"""

import streamlit as st
from utils.document_loader import DocumentLoader
from utils.retriever import EmbeddingRetriever
from utils.generator import AnswerGenerator


# Page configuration
st.set_page_config(
    page_title="RAG Data Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .section-header {
        color: #1f77b4;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📚 RAG Data Assistant</div>', unsafe_allow_html=True)
st.markdown("Upload a document and ask questions about its content using AI-powered retrieval.")

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = EmbeddingRetriever()
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Sidebar
st.sidebar.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 1000, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", 0, 100, 50, step=10)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-header">📤 Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a document to upload",
        type=["txt", "pdf", "csv"],
        help="Upload a .txt, .pdf, or .csv file to process"
    )
    
    if uploaded_file is not None:
        # Read and process the document
        with st.spinner("Processing document..."):
            try:
                # Load and chunk the document
                loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                text_content = loader.load_file(uploaded_file)
                chunks = loader.chunk_text(text_content)

                if not chunks:
                    raise ValueError("No readable text was found in this file.")

                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.document_loaded = True

                # Build retriever index
                st.session_state.retriever.build_index(chunks)
            except Exception as error:
                st.session_state.document_loaded = False
                st.session_state.chunks = []
                st.error(f"Could not process this file. {error}")

        if st.session_state.document_loaded:
            st.success(f"✅ Document processed! Created {len(chunks)} chunks.")
            st.info(f"File: {uploaded_file.name} | Size: {len(text_content):,} characters")

with col2:
    st.markdown('<div class="section-header">📊 Document Statistics</div>', unsafe_allow_html=True)
    
    if st.session_state.document_loaded:
        total_chars = sum(len(chunk) for chunk in st.session_state.chunks)
        avg_chunk_size = total_chars / len(st.session_state.chunks) if st.session_state.chunks else 0
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Chunks", len(st.session_state.chunks))
        with col_stat2:
            st.metric("Total Characters", f"{total_chars:,}")
        with col_stat3:
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f}")
    else:
        st.info("Upload a document to see statistics")

# Question answering section
st.markdown('<div class="section-header">❓ Ask a Question</div>', unsafe_allow_html=True)

if st.session_state.document_loaded:
    query = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        label_visibility="collapsed"
    )
    
    col_submit, col_clear = st.columns([3, 1])
    
    with col_submit:
        if st.button("🔍 Search & Generate Answer", use_container_width=True):
            if query.strip():
                with st.spinner("Retrieving relevant information..."):
                    # Retrieve relevant chunks
                    results = st.session_state.retriever.retrieve(query, k=3)
                    
                    # Generate answer
                    answer = AnswerGenerator.generate_answer(query, results)
                    formatted_context = AnswerGenerator.format_retrieval_results(query, results)
                
                # Display results
                st.markdown('<div class="section-header">📋 Retrieved Context</div>', unsafe_allow_html=True)
                st.markdown(formatted_context)
                
                st.markdown('<div class="section-header">💭 Generated Answer</div>', unsafe_allow_html=True)
                st.markdown(answer)
            else:
                st.warning("Please enter a question.")
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.document_loaded = False
            st.session_state.chunks = []
            st.session_state.retriever = EmbeddingRetriever()
            st.rerun()

else:
    st.info("👆 Upload a document first to start asking questions!")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em; margin-top: 20px;">
        <p>RAG Data Assistant | Powered by Streamlit, Sentence-Transformers, and FAISS</p>
        <p>Built as a portfolio project | No API keys required</p>
    </div>
""", unsafe_allow_html=True)
