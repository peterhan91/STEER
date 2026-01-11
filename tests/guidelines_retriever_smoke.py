#!/usr/bin/env python3
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from guidelines.retriever import (
    iter_guideline_docs,
    build_bm25_retriever,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test guideline retrieval with BM25."
    )
    parser.add_argument(
        "--guidelines",
        default=os.path.join("guidelines", "open_guidelines.jsonl"),
        help="Path to guidelines JSONL.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text (e.g., patient history).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2000,
        help="Max JSONL lines to scan; set to -1 for all.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1200, help="Chunk size in characters."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150, help="Chunk overlap in characters."
    )
    parser.add_argument("--k", type=int, default=4, help="Top-k results to return.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="Preview length for each result.",
    )
    parser.add_argument(
        "--source-filter",
        default="",
        help="Optional substring filter on the guideline source.",
    )
    args = parser.parse_args()

    max_lines = None if args.max_lines < 0 else args.max_lines
    docs = list(iter_guideline_docs(args.guidelines, max_lines, args.source_filter))
    if not docs:
        print("No guideline documents found.", file=sys.stderr)
        return 1

    retriever = build_bm25_retriever(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, k=args.k
    )
    results = retriever.get_relevant_documents(args.query)

    print(f"Loaded {len(docs)} guideline docs.")
    print(f"Top {len(results)} results:")
    for idx, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        title = meta.get("title") or "unknown"
        source = meta.get("source") or "unknown"
        preview = doc.page_content.replace("\n", " ")
        if len(preview) > args.preview_chars:
            preview = preview[: args.preview_chars].rstrip() + "..."
        print(f"{idx}. source={source} title={title}")
        print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
