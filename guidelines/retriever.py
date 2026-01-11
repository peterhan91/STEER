from typing import Iterable, List, Optional
import json

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever


def iter_guideline_docs(
    path: str, max_lines: Optional[int] = None, source_filter: Optional[str] = None
) -> Iterable[Document]:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if max_lines is not None and max_lines >= 0 and count >= max_lines:
                break
            count += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("clean_text") or obj.get("raw_text") or ""
            if not text:
                continue
            source = (obj.get("source") or "").strip()
            if source_filter and source_filter not in source:
                continue
            meta = {
                "id": obj.get("id"),
                "source": source,
                "title": obj.get("title"),
            }
            yield Document(page_content=text, metadata=meta)


def build_bm25_retriever(
    docs: List[Document],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    k: int = 4,
) -> BM25Retriever:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever


def retrieve_guideline_snippets(
    *,
    path: str,
    query: str,
    max_lines: Optional[int] = 2000,
    source_filter: Optional[str] = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    k: int = 4,
) -> List[Document]:
    docs = list(iter_guideline_docs(path, max_lines, source_filter))
    if not docs:
        return []
    retriever = build_bm25_retriever(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
    )
    return retriever.get_relevant_documents(query)
