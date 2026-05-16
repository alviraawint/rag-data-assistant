"""
Generator module for creating answers based on retrieved context.
"""

from typing import List, Tuple


class AnswerGenerator:
    """Generate answers based on retrieved context."""

    @staticmethod
    def generate_answer(query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """
        Generate a simple answer based on retrieved context.
        
        This is a template-based approach that doesn't require external APIs.
        For more advanced generation, you could integrate a local LLM like Ollama.
        
        Args:
            query: The user's question
            context_chunks: List of (chunk, score) tuples from retrieval
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "I couldn't find relevant information to answer your question. Please try uploading a document first."
        
        # Combine the most relevant chunks as context
        context = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        # Create a simple template-based answer
        answer = f"""Based on the retrieved context from your document:

{context}

---

**Answer:** The document contains information relevant to your question "{query}". The most relevant sections are shown above. You can review them to find the specific answer you're looking for.
"""
        return answer

    @staticmethod
    def format_retrieval_results(
        query: str, 
        context_chunks: List[Tuple[str, float]]
    ) -> str:
        """
        Format retrieval results for display.
        
        Args:
            query: The user's question
            context_chunks: List of (chunk, score) tuples
            
        Returns:
            Formatted string for display
        """
        if not context_chunks:
            return "No relevant chunks found."
        
        formatted = "**Retrieved Context (Top 3 Chunks):**\n\n"
        for i, (chunk, score) in enumerate(context_chunks, 1):
            formatted += f"**Chunk {i}** (Relevance: {score:.2%})\n"
            formatted += f"{chunk}\n\n"
            formatted += "---\n\n"
        
        return formatted
