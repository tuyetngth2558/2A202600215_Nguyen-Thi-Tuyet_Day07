from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        normalized = text.strip()
        if not normalized:
            return []

        # A simple, robust sentence extractor. Keeps punctuation at end when present.
        # Works for: ". ", "! ", "? ", and also handles newlines.
        sentences = [
            s.strip()
            for s in re.findall(r"[^.!?\n]+[.!?]?(?:\s+|\n+|$)", normalized)
            if s.strip()
        ]

        chunks: list[str] = []
        current: list[str] = []
        for s in sentences:
            current.append(s)
            if len(current) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current).strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        return [c for c in chunks if c]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        t = text.strip()
        if not t:
            return []
        return [c for c in self._split(t, list(self.separators)) if c]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        t = current_text.strip()
        if not t:
            return []
        if len(t) <= self.chunk_size:
            return [t]

        if not remaining_separators:
            # Fall back: hard cut into fixed windows.
            return [t[i : i + self.chunk_size].strip() for i in range(0, len(t), self.chunk_size) if t[i : i + self.chunk_size].strip()]

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # If sep is empty string, we can't split further by delimiter; hard cut.
        if sep == "":
            return [t[i : i + self.chunk_size].strip() for i in range(0, len(t), self.chunk_size) if t[i : i + self.chunk_size].strip()]

        if sep not in t:
            return self._split(t, rest)

        parts = [p for p in t.split(sep) if p is not None and p != ""]

        # First, recursively reduce any oversized parts.
        reduced_parts: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > self.chunk_size:
                reduced_parts.extend(self._split(p, rest))
            else:
                reduced_parts.append(p)

        # Then, merge adjacent parts back up to chunk_size.
        merged: list[str] = []
        buf = ""
        joiner = sep
        for p in reduced_parts:
            candidate = p if not buf else f"{buf}{joiner}{p}"
            if len(candidate) <= self.chunk_size:
                buf = candidate
            else:
                if buf:
                    merged.append(buf.strip())
                buf = p
        if buf:
            merged.append(buf.strip())

        # Some merged chunks might still exceed chunk_size (edge cases); reduce again.
        final: list[str] = []
        for m in merged:
            if len(m) > self.chunk_size:
                final.extend(self._split(m, rest))
            else:
                final.append(m)
        return [c for c in final if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    denom = (math.sqrt(_dot(vec_a, vec_a)) * math.sqrt(_dot(vec_b, vec_b)))
    if denom == 0:
        return 0.0
    return _dot(vec_a, vec_b) / denom


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
