"""
Retriever module for creating embeddings and searching using FAISS.
"""

import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

# This app uses sentence-transformers through PyTorch. If TensorFlow is
# installed globally, transformers may try to import it and hit unrelated
# TensorFlow/protobuf version conflicts before the app starts.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"),
)

from sentence_transformers import SentenceTransformer


MODEL_CACHE = Path(os.environ["HF_HOME"]) / "hub"


def _is_model_cached(model_name: str) -> bool:
    model_names = [model_name]
    if "/" not in model_name:
        model_names.append(f"sentence-transformers/{model_name}")

    for name in model_names:
        model_cache_name = f"models--{name.replace('/', '--')}"
        snapshots_dir = MODEL_CACHE / model_cache_name / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True
    return False


class EmbeddingRetriever:
    """Retrieve relevant chunks using embeddings and FAISS."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        local_files_only = _is_model_cached(model_name)
        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(MODEL_CACHE),
                local_files_only=local_files_only,
            )
        except OSError:
            if not local_files_only:
                raise
            self.model = SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE))
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
