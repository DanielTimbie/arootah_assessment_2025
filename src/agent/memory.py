from __future__ import annotations
import sqlite3, hashlib, pickle, time
from typing import List, Tuple, Optional
import numpy as np
from .config import settings

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
    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.sqlite_path
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def upsert(self, kind: str, url: str, title: str, snippet: str, content: str, embedding: Optional[np.ndarray]) -> Optional[int]:
        h = self._hash(content or (title + snippet))
        emb_blob = pickle.dumps(embedding.astype(np.float32)) if embedding is not None else None
        try:
            cur = self.conn.execute(
                "INSERT INTO documents(kind,url,title,snippet,content,content_hash,embedding,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (kind, url, title, snippet, content, h, emb_blob, time.time()),
            )
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def fetch_all(self) -> List[Tuple[int, str, str, str, str, str, Optional[bytes]]]:
        cur = self.conn.execute("SELECT id, kind, url, title, snippet, content, embedding FROM documents ORDER BY id ASC")
        return cur.fetchall()

    def nearest(self, embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        rows = self.conn.execute("SELECT id, embedding FROM documents").fetchall()
        sims = []
        for rid, blob in rows:
            if blob is None:
                continue
            vec = pickle.loads(blob)
            denom = (np.linalg.norm(vec) * np.linalg.norm(embedding))
            if denom == 0:
                continue
            sim = float(np.dot(vec, embedding) / denom)
            sims.append((rid, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def is_duplicate(self, embedding: np.ndarray, threshold: Optional[float] = None) -> bool:
        threshold = threshold or settings.sim_threshold
        sims = self.nearest(embedding, k=3)
        return any(sim >= threshold for _, sim in sims)