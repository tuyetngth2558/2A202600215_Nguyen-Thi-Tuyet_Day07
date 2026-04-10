from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
    "20K-AI-Handbook_final.pdf",
]

try:
    # Avoid Windows console UnicodeEncodeError on some setups.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass



def _read_pdf_text(path: Path) -> str:
    """
    Best-effort PDF to text extraction for manual demo usage.

    Tries (in order):
      1) pymupdf4llm (higher quality markdown-ish)
      2) PyMuPDF (fitz) plain text extraction
    """
    try:
        import pymupdf4llm  # type: ignore

        return pymupdf4llm.to_markdown(str(path))
    except Exception:
        pass

    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(
            "PDF support is not installed. Install one of:\n"
            "  pip install pymupdf4llm\n"
            "  pip install pymupdf\n"
            f"\nOriginal error: {e}"
        ) from e


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt", ".pdf"}
    documents: list[Document] = []
    chunker = RecursiveChunker(chunk_size=1200)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        if path.suffix.lower() == ".pdf":
            try:
                content = _read_pdf_text(path)
            except Exception as e:
                print(f"Skipping PDF (failed to read): {path}\n  reason: {e}")
                continue
        else:
            content = path.read_text(encoding="utf-8")

        content = content.strip()
        if not content:
            print(f"Skipping empty file: {path}")
            continue

        # Chunk large sources (especially PDFs) so retrieval works better.
        chunks = chunker.chunk(content) if len(content) > 1400 else [content]
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}__chunk{i:04d}",
                    content=chunk,
                    metadata={
                        "source": str(path),
                        "extension": path.suffix.lower(),
                        "doc_id": path.stem,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple extractive mock LLM for manual RAG testing."""
    try:
        context_marker = "Context:\n"
        question_marker = "\n\nQuestion:"
        answer_marker = "\nAnswer:"

        context_start = prompt.index(context_marker) + len(context_marker)
        question_start = prompt.index(question_marker)
        answer_start = prompt.index(answer_marker)

        context = prompt[context_start:question_start].strip()
        question = prompt[question_start + len(question_marker) : answer_start].strip().lower()

        question_tokens = [t for t in question.replace("?", " ").split() if len(t) > 2]
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        sentences: list[str] = []
        for line in lines:
            parts = [p.strip() for p in line.replace("!", ".").replace("?", ".").split(".") if p.strip()]
            sentences.extend(parts)

        # Heuristic for definition-style questions: "<subject> là gì?"
        subject = question.replace("là gì", "").replace("?", "").strip()
        if subject:
            for sent in sentences:
                s = sent.lower()
                if subject in s and " là " in f" {s} ":
                    return f"[DEMO ANSWER] {sent}."

        best_line = ""
        best_score = -1
        for line in sentences or lines:
            line_lower = line.lower()
            score = sum(1 for t in question_tokens if t in line_lower)
            if score > best_score:
                best_score = score
                best_line = line

        if best_line:
            return f"[DEMO ANSWER] {best_line}"
        return "[DEMO ANSWER] Không tìm thấy thông tin phù hợp trong context."
    except Exception:
        preview = prompt[:300].replace("\n", " ")
        return f"[DEMO ANSWER] Không phân tích được prompt. Preview: {preview}..."


def run_manual_demo(
    question: str | None = None,
    sample_files: list[str] | None = None,
    source_filter: str | None = None,
) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt, .pdf")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    metadata_filter = {"source": source_filter} if source_filter else None
    search_results = (
        store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)
        if metadata_filter
        else store.search(query, top_k=3)
    )
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    if source_filter:
        print(f"Source filter: {source_filter}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3, metadata_filter=metadata_filter))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual RAG demo")
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument(
        "--files",
        default="",
        help="Comma-separated file list. Example: --files \"Giao trinh Triet hoc.md\"",
    )
    parser.add_argument(
        "--source-filter",
        default="",
        help="Exact source path in metadata to restrict retrieval, e.g. Giao trinh Triet hoc.md",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() if args.question else None
    files = [f.strip() for f in args.files.split(",") if f.strip()] if args.files else None
    source_filter = args.source_filter.strip() or None
    return run_manual_demo(question=question, sample_files=files, source_filter=source_filter)


if __name__ == "__main__":
    raise SystemExit(main())
