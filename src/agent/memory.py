"""sqlite-based memory for deduplication."""
from __future__ import annotations

import hashlib
import pickle
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from .config import settings


@dataclass
class Document:
    """Represents a stored document with metadata and content."""

    id: int | None
    kind: str
    url: str
    title: str
    snippet: str
    content: str
    content_hash: str | None = None
    embedding: np.ndarray | None = None
    created_at: float | None = None

@dataclass
class SimilarityResult:
    """Represents a similarity search result from memory."""

    document_id: int
    similarity_score: float

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  url TEXT NOT NULL,
  title TEXT,
  snippet TEXT,
  content TEXT,
  content_hash TEXT UNIQUE,
  embedding BLOB,
  created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
"""

class SqliteMemory:
    """sqlite storage for embeddings and deduplication."""

    def __init__(self, path: str | None = None) -> None:
        """initialize sqlite connection and create tables."""
        self.path = path or settings.sqlite_path
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)

    @staticmethod
    def _hash(text: str) -> str:
        """generate sha256 hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def upsert(self, document: Document) -> int | None:
        """insert document with embedding, skip if duplicate hash."""
        content_for_hash = document.content or (document.title + document.snippet)
        h = self._hash(content_for_hash)
        emb_blob = (
            pickle.dumps(document.embedding.astype(np.float32))
            if document.embedding is not None
            else None
        )
        try:
            cur = self.conn.execute(
                "INSERT INTO documents(kind,url,title,snippet,content,content_hash,"
                "embedding,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (document.kind, document.url, document.title, document.snippet,
                 document.content, h, emb_blob, time.time()),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            return None
        else:
            self.conn.commit()
            return cur.lastrowid

    def fetch_all(self) -> list[Document]:
        """fetch all stored documents."""
        cur = self.conn.execute(
            "SELECT id, kind, url, title, snippet, content, embedding, created_at "
            "FROM documents ORDER BY id ASC"
        )
        documents = []
        for row in cur.fetchall():
            doc_id, kind, url, title, snippet, content, embedding_blob, created_at = row
            embedding = pickle.loads(embedding_blob) if embedding_blob else None
            documents.append(Document(
                id=doc_id,
                kind=kind,
                url=url,
                title=title,
                snippet=snippet,
                content=content,
                embedding=embedding,
                created_at=created_at
            ))
        return documents

    def nearest(self, embedding: np.ndarray, k: int = 5) -> list[SimilarityResult]:
        """find k nearest embeddings by cosine similarity."""
        rows = self.conn.execute("SELECT id, embedding FROM documents").fetchall()
        results = []
        for document_id, blob in rows:
            if blob is None:
                continue
            vec = pickle.loads(blob)
            denom = (np.linalg.norm(vec) * np.linalg.norm(embedding))
            if denom == 0:
                continue
            sim = float(np.dot(vec, embedding) / denom)
            results.append(
                SimilarityResult(document_id=document_id, similarity_score=sim)
            )
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:k]

    def is_duplicate(
        self, embedding: np.ndarray, threshold: float | None = None
    ) -> bool:
        """check if embedding is duplicate above threshold."""
        threshold = threshold or settings.sim_threshold
        results = self.nearest(embedding, k=3)
        return any(result.similarity_score >= threshold for result in results)
