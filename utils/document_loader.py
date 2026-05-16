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
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Target number of overlapping characters between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

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
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into sentence-aware chunks with configurable overlap.

        The method first keeps sentence boundaries where possible, then carries
        the last few sentences from the previous chunk into the next chunk until
        the configured overlap budget is reached. If a single sentence is longer
        than ``chunk_size``, it is split by characters as a fallback.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        text = self.load_text(text)
        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences).strip())
                    current_sentences = self._overlap_sentences(current_sentences)
                    current_length = self._joined_length(current_sentences)

                long_sentence_chunks = self._split_long_sentence(sentence)
                chunks.extend(long_sentence_chunks[:-1])
                current_sentences = [long_sentence_chunks[-1]] if long_sentence_chunks else []
                current_length = self._joined_length(current_sentences)
                continue

            separator_length = 1 if current_sentences else 0
            next_length = current_length + separator_length + len(sentence)

            if current_sentences and next_length > self.chunk_size:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = self._overlap_sentences(current_sentences)
                current_length = self._joined_length(current_sentences)

                while current_sentences:
                    separator_length = 1 if current_sentences else 0
                    next_length = current_length + separator_length + len(sentence)
                    if next_length <= self.chunk_size:
                        break
                    current_sentences.pop(0)
                    current_length = self._joined_length(current_sentences)

            current_sentences.append(sentence)
            current_length = self._joined_length(current_sentences)

        if current_sentences:
            final_chunk = " ".join(current_sentences).strip()
            if final_chunk and (not chunks or final_chunk != chunks[-1]):
                chunks.append(final_chunk)

        return chunks

    def _overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Return trailing sentences to reuse in the next chunk."""
        if self.chunk_overlap == 0:
            return []

        overlap = []
        overlap_length = 0
        for sentence in reversed(sentences):
            separator_length = 1 if overlap else 0
            next_length = overlap_length + separator_length + len(sentence)

            if overlap and next_length > self.chunk_overlap:
                break

            overlap.insert(0, sentence)
            overlap_length = self._joined_length(overlap)

            if overlap_length >= self.chunk_overlap:
                break

        return overlap

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into character windows with overlap."""
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        start = 0

        while start < len(sentence):
            end = start + self.chunk_size
            chunks.append(sentence[start:end].strip())
            if end >= len(sentence):
                break
            start += step

        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _joined_length(sentences: List[str]) -> int:
        """Calculate the length of sentences joined with a single space."""
        if not sentences:
            return 0
        return sum(len(sentence) for sentence in sentences) + len(sentences) - 1
