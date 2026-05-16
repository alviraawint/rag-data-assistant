"""
Generator module for creating answers based on retrieved context.
"""

import re
from collections import Counter
from typing import List, Tuple


class AnswerGenerator:
    """Generate answers based on retrieved context."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "how", "i", "in", "is", "it", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when", "where",
        "which", "who", "why", "with", "you", "your",
    }

    SKILL_ALIASES = {
        "python": "Python",
        "sql": "SQL",
        "excel": "Excel",
        "power bi": "Power BI",
        "tableau": "Tableau",
        "data analysis": "Data analysis",
        "data analytics": "Data analytics",
        "data science": "Data science",
        "machine learning": "Machine learning",
        "deep learning": "Deep learning",
        "natural language processing": "Natural language processing",
        "nlp": "NLP",
        "rag": "RAG",
        "streamlit": "Streamlit",
        "faiss": "FAISS",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "scikit-learn": "Scikit-learn",
        "sklearn": "Scikit-learn",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "statistics": "Statistics",
        "visualization": "Data visualization",
        "data visualization": "Data visualization",
        "dashboard": "Dashboard development",
        "dashboards": "Dashboard development",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "react": "React",
        "html": "HTML",
        "css": "CSS",
        "flask": "Flask",
        "django": "Django",
        "fastapi": "FastAPI",
        "docker": "Docker",
        "git": "Git",
        "github": "GitHub",
        "aws": "AWS",
        "azure": "Azure",
        "gcp": "Google Cloud",
        "java": "Java",
        "c++": "C++",
        "communication": "Communication",
        "leadership": "Leadership",
        "project management": "Project management",
        "problem solving": "Problem solving",
    }

    @staticmethod
    def generate_answer(query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """
        Generate a concise answer based on retrieved context.
        
        This rule-based approach doesn't require external APIs. It selects
        relevant sentences from retrieved chunks and has a special path for
        resume skill questions.
        
        Args:
            query: The user's question
            context_chunks: List of (chunk, score) tuples from retrieval
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "I couldn't find relevant information to answer your question. Please try uploading a document first."

        if AnswerGenerator._is_skill_question(query):
            return AnswerGenerator._generate_skill_answer(context_chunks)

        summary = AnswerGenerator._extract_relevant_summary(query, context_chunks)
        return f"Based on the document, {summary}"

    @staticmethod
    def _is_skill_question(query: str) -> bool:
        query = query.lower()
        skill_phrases = (
            "top skill",
            "top skills",
            "main skill",
            "main skills",
            "strongest skill",
            "strongest skills",
        )
        return any(phrase in query for phrase in skill_phrases)

    @staticmethod
    def _generate_skill_answer(context_chunks: List[Tuple[str, float]]) -> str:
        skill_scores = AnswerGenerator._score_skills(context_chunks)

        if skill_scores:
            top_skills = [skill for skill, _ in skill_scores.most_common(4)]
            strongest_skill = top_skills[0]
            supporting_skills = top_skills[1:]

            if supporting_skills:
                support_text = ", ".join(supporting_skills)
                return (
                    "Based on the document, the strongest skill appears to be "
                    f"{strongest_skill}. The document also highlights related "
                    f"strengths in {support_text}."
                )

            return (
                "Based on the document, the strongest skill appears to be "
                f"{strongest_skill}."
            )

        summary = AnswerGenerator._extract_relevant_summary("skills experience", context_chunks)
        return f"Based on the document, {summary}"

    @staticmethod
    def _score_skills(context_chunks: List[Tuple[str, float]]) -> Counter:
        skill_scores = Counter()

        for chunk, relevance in context_chunks:
            lower_chunk = chunk.lower()
            weight = max(relevance, 0.1)

            for alias, skill_name in AnswerGenerator.SKILL_ALIASES.items():
                pattern = r"\b" + re.escape(alias) + r"\b"
                matches = re.findall(pattern, lower_chunk)
                if matches:
                    skill_scores[skill_name] += len(matches) * weight

            for skill in AnswerGenerator._extract_skills_from_labeled_sections(chunk):
                skill_scores[skill] += 2 * weight

        return skill_scores

    @staticmethod
    def _extract_skills_from_labeled_sections(text: str) -> List[str]:
        skills = []
        section_matches = re.findall(
            r"(?:technical skills|core skills|skills|tools|technologies)\s*[:\-]\s*([^.\n]+)",
            text,
            flags=re.IGNORECASE,
        )

        for section in section_matches:
            parts = re.split(r",|;|\||/|\band\b", section)
            for part in parts:
                skill = part.strip(" -:()[]{}")
                if AnswerGenerator._looks_like_skill(skill):
                    normalized_skill = skill.lower()
                    skills.append(
                        AnswerGenerator.SKILL_ALIASES.get(normalized_skill, skill)
                    )

        return skills

    @staticmethod
    def _looks_like_skill(text: str) -> bool:
        if not text:
            return False

        words = text.split()
        if len(words) > 4:
            return False

        rejected_words = {"experience", "education", "project", "projects", "work"}
        return not any(word.lower() in rejected_words for word in words)

    @staticmethod
    def _extract_relevant_summary(
        query: str,
        context_chunks: List[Tuple[str, float]],
        max_sentences: int = 2,
    ) -> str:
        query_terms = AnswerGenerator._important_words(query)
        scored_sentences = []

        for chunk, relevance in context_chunks:
            sentences = AnswerGenerator._split_sentences(chunk)
            for sentence in sentences:
                sentence_terms = AnswerGenerator._important_words(sentence)
                overlap = len(query_terms.intersection(sentence_terms))
                score = overlap + relevance
                scored_sentences.append((score, sentence.strip()))

        if not scored_sentences:
            return AnswerGenerator._shorten(context_chunks[0][0])

        scored_sentences.sort(key=lambda item: item[0], reverse=True)

        chosen = []
        seen = set()
        for _, sentence in scored_sentences:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            chosen.append(AnswerGenerator._shorten(sentence))
            seen.add(normalized)
            if len(chosen) == max_sentences:
                break

        if any(sentence.lower().startswith("row ") for sentence in chosen):
            chosen = [
                sentence
                for sentence in chosen
                if not sentence.lower().startswith("columns:")
            ]

        return " ".join(chosen)

    @staticmethod
    def _important_words(text: str) -> set:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text.lower())
        return {
            word
            for word in words
            if word not in AnswerGenerator.STOP_WORDS and len(word) > 2
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sections = re.split(r"\s+(?=Row \d+:|Page \d+|[A-Z][A-Za-z ]{2,}:)", text)
        sentences = []
        for section in sections:
            sentences.extend(re.split(r"(?<=[.!?])\s+", section))

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    @staticmethod
    def _shorten(text: str, max_length: int = 300) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_length:
            return text

        shortened = text[:max_length].rsplit(" ", 1)[0]
        return f"{shortened}..."

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
