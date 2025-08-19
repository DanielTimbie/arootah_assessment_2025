"""test memory storage and deduplication."""
import numpy as np

from src.agent.memory import Document, SqliteMemory


def test_upsert_and_duplicate():
    """test document storage and duplicate detection."""
    m = SqliteMemory(":memory:")
    v = np.ones(3, dtype=np.float32)
    doc = Document(
        id=None, kind="test", url="http://test.com",
        title="test", snippet="test", content="test", embedding=v
    )
    doc_id = m.upsert(doc)
    assert doc_id is not None
    assert m.is_duplicate(v, threshold=0.5) is True
