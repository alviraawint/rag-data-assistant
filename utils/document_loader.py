"""
Document loader module for splitting documents into chunks.
"""

import io
import re
from typing import List

import pandas as pd
from pypdf import PdfReader


class DocumentLoader:
    """Load and chunk documents for RAG."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, uploaded_file) -> str:
        """
        Load text from a Streamlit uploaded file.

        Args:
            uploaded_file: File uploaded through st.file_uploader

        Returns:
            Text extracted from the file
        """
        file_name = uploaded_file.name
        file_type = file_name.rsplit(".", 1)[-1].lower()
        file_bytes = uploaded_file.getvalue()

        try:
            if file_type == "txt":
                return self.load_txt(file_bytes)
            if file_type == "pdf":
                return self.load_pdf(file_bytes)
            if file_type == "csv":
                return self.load_csv(file_bytes)
        except Exception as error:
            raise ValueError(f"Could not process {file_name}: {error}") from error

        raise ValueError(f"Unsupported file type: .{file_type}")

    def load_txt(self, file_bytes: bytes) -> str:
        """Read plain text from a TXT file."""
        text = file_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValueError("The text file is empty.")
        return text

    def load_pdf(self, file_bytes: bytes) -> str:
        """Extract text from every page in a PDF file."""
        reader = PdfReader(io.BytesIO(file_bytes))
        page_texts = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(f"Page {page_number}\n{text.strip()}")

        if not page_texts:
            raise ValueError("No readable text was found in this PDF.")

        return "\n\n".join(page_texts)

    def load_csv(self, file_bytes: bytes) -> str:
        """Convert CSV rows into readable text with column names."""
        dataframe = pd.read_csv(io.BytesIO(file_bytes))

        if len(dataframe.columns) == 0:
            raise ValueError("The CSV file has no columns.")

        lines = [f"Columns: {', '.join(dataframe.columns)}"]

        if dataframe.empty:
            lines.append("No rows found.")
            return "\n".join(lines)

        for row_number, (_, row) in enumerate(dataframe.iterrows(), start=1):
            row_values = []
            for column in dataframe.columns:
                value = row[column]
                if pd.isna(value):
                    value = ""
                row_values.append(f"{column}: {value}")
            lines.append(f"Row {row_number}: " + "; ".join(row_values))

        return "\n".join(lines)

    def load_text(self, text: str) -> str:
        """
        Load and clean text from document.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        text = self.load_text(text)
        
        # Split by sentences first for better context
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
