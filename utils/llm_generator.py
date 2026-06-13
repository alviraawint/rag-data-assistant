"""
LLM-backed answer generation grounded in retrieved context.

This module turns the project into a *real* Retrieval-Augmented Generation
system: retrieved chunks are passed to Claude as context, and the model is
instructed to answer using only that context (with inline citations). When no
API key is configured, the app falls back to the rule-based AnswerGenerator so
it still runs with zero external dependencies.
"""

import os
from typing import Iterator, List, Tuple

try:
    import anthropic
except ImportError:  # pragma: no cover - anthropic is an optional dependency
    anthropic = None


# Default model. Override with the RAG_LLM_MODEL environment variable.
# claude-opus-4-8 is the most capable model; for a cheaper / faster demo set
# RAG_LLM_MODEL=claude-haiku-4-5.
DEFAULT_MODEL = "claude-opus-4-8"

# A grounded prompt is what separates RAG from "ask a chatbot". The model is
# told to stay strictly within the retrieved passages and to refuse when the
# answer is not present, which is the core hallucination-control mechanism.
SYSTEM_PROMPT = (
    "You are a precise question-answering assistant for a retrieval-augmented "
    "generation (RAG) system. Answer the user's question using ONLY the "
    "numbered context passages provided. Follow these rules strictly:\n"
    "- Base every statement on the context. Do not use outside knowledge.\n"
    "- Cite the supporting passage inline as [Chunk N] after each claim.\n"
    "- If the context does not contain the answer, reply exactly: "
    '"The document does not contain information to answer this question." '
    "Do not guess or invent details.\n"
    "- Be concise: 2-5 sentences. Do not restate the question."
)

NO_CONTEXT_ANSWER = (
    "The document does not contain information to answer this question."
)


class ClaudeAnswerGenerator:
    """Generate grounded answers from retrieved context using the Claude API."""

    def __init__(self, model: str = None):
        """
        Args:
            model: Claude model id. Defaults to the RAG_LLM_MODEL environment
                variable, then to DEFAULT_MODEL.
        """
        if anthropic is None:
            raise RuntimeError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            )
        self.model = model or os.environ.get("RAG_LLM_MODEL", DEFAULT_MODEL)
        # The SDK reads ANTHROPIC_API_KEY from the environment automatically.
        self.client = anthropic.Anthropic()

    @staticmethod
    def is_available() -> bool:
        """True only if the SDK is installed and an API key is configured."""
        return anthropic is not None and bool(os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def _build_user_prompt(
        query: str, context_chunks: List[Tuple[str, float]]
    ) -> str:
        """Assemble the retrieved chunks and question into a single prompt."""
        lines = ["Context passages:", ""]
        for index, (chunk, score) in enumerate(context_chunks, start=1):
            lines.append(f"[Chunk {index}] (relevance {score:.0%})")
            lines.append(chunk.strip())
            lines.append("")
        lines.append(f"Question: {query.strip()}")
        return "\n".join(lines)

    def stream_answer(
        self, query: str, context_chunks: List[Tuple[str, float]]
    ) -> Iterator[str]:
        """
        Stream a grounded answer token-by-token.

        Yields text deltas so the UI can render the answer as it is produced.
        Raises anthropic.* errors on API failure so the caller can fall back.
        """
        if not context_chunks:
            yield NO_CONTEXT_ANSWER
            return

        user_prompt = self._build_user_prompt(query, context_chunks)

        # Streaming keeps the UI responsive and avoids request timeouts.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def generate_answer(
        self, query: str, context_chunks: List[Tuple[str, float]]
    ) -> str:
        """Return the full grounded answer as a single string."""
        return "".join(self.stream_answer(query, context_chunks))
