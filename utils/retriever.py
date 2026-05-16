"""
Retriever module for creating embeddings and searching using FAISS.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class EmbeddingRetriever:
    """Retrieve relevant chunks using embeddings and FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return embeddings

    def build_index(self, chunks: List[str]) -> None:
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of text chunks
        """
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Query string
            k: Number of top chunks to retrieve
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []
        
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                # Convert L2 distance to similarity score (lower distance = higher relevance)
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))
        
        return results

    def is_empty(self) -> bool:
        """Check if the retriever has any indexed chunks."""
        return self.index is None or len(self.chunks) == 0
