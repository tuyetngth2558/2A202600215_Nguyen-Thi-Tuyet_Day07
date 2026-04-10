from __future__ import annotations

import re
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            from chromadb import Client  # type: ignore

            client = Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        record_id = f"{doc.id}::{self._next_index}"
        self._next_index += 1

        # Ensure each stored item has doc_id in metadata for delete/filter tests.
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)

        return {
            "id": record_id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        q_emb = self._embedding_fn(query)
        q_tokens = set(re.findall(r"\w+", query.lower()))
        query_norm = " ".join(re.findall(r"\w+", query.lower()))
        scored: list[dict[str, Any]] = []
        for r in records:
            semantic_score = _dot(q_emb, r["embedding"])
            doc_text = str(r.get("content", "")).lower()
            doc_tokens = set(re.findall(r"\w+", doc_text))
            doc_norm = " ".join(re.findall(r"\w+", doc_text))
            overlap = len(q_tokens & doc_tokens)
            lexical_score = (overlap / max(1, len(q_tokens))) if q_tokens else 0.0
            phrase_bonus = 1.0 if query_norm and query_norm in doc_norm else 0.0
            # Hybrid score helps when using mock embeddings in classroom demos.
            score = semantic_score + 1.0 * lexical_score + phrase_bonus
            scored.append(
                {
                    "id": r.get("id"),
                    "content": r.get("content"),
                    "metadata": r.get("metadata", {}),
                    "score": score,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict[str, Any]] = []
            for d in docs:
                rec = self._make_record(d)
                ids.append(rec["id"])
                documents.append(rec["content"])
                embeddings.append(rec["embedding"])
                metadatas.append(rec["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            return

        for d in docs:
            self._store.append(self._make_record(d))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            q_emb = self._embedding_fn(query)
            out = self._collection.query(query_embeddings=[q_emb], n_results=top_k)
            results: list[dict[str, Any]] = []
            docs = (out.get("documents") or [[]])[0]
            metas = (out.get("metadatas") or [[]])[0]
            dists = (out.get("distances") or [[]])[0]
            ids = (out.get("ids") or [[]])[0]
            for i in range(len(docs)):
                # Convert distance-like to score-like (higher is better). If chroma returns cosine distance,
                # score = 1 - distance is a reasonable mapping.
                dist = float(dists[i]) if dists and i < len(dists) else 0.0
                score = 1.0 - dist
                results.append({"id": ids[i], "content": docs[i], "metadata": metas[i] or {}, "score": score})
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            try:
                return int(self._collection.count())
            except Exception:
                return 0
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        def _match(meta: dict[str, Any]) -> bool:
            for k, v in metadata_filter.items():
                if meta.get(k) != v:
                    return False
            return True

        if self._use_chroma and self._collection is not None:
            # Chroma supports where filters; keep a safe fallback if it errors.
            try:
                q_emb = self._embedding_fn(query)
                out = self._collection.query(query_embeddings=[q_emb], n_results=top_k, where=metadata_filter)
                results: list[dict[str, Any]] = []
                docs = (out.get("documents") or [[]])[0]
                metas = (out.get("metadatas") or [[]])[0]
                dists = (out.get("distances") or [[]])[0]
                ids = (out.get("ids") or [[]])[0]
                for i in range(len(docs)):
                    dist = float(dists[i]) if dists and i < len(dists) else 0.0
                    score = 1.0 - dist
                    results.append({"id": ids[i], "content": docs[i], "metadata": metas[i] or {}, "score": score})
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
            except Exception:
                pass

        filtered = [r for r in self._store if _match(r.get("metadata", {}))]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        removed_any = False

        if self._use_chroma and self._collection is not None:
            try:
                # If metadatas contain doc_id, we can delete via where filter.
                self._collection.delete(where={"doc_id": doc_id})
                removed_any = True
            except Exception:
                # Fall back to in-memory state only.
                removed_any = False

        before = len(self._store)
        self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
        if len(self._store) != before:
            removed_any = True

        return removed_any
