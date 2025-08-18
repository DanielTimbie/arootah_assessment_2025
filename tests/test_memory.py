from src.agent.memory import SqliteMemory
import numpy as np

def test_upsert_and_duplicate():
    m = SqliteMemory(":memory:")
    v = np.ones(3, dtype=np.float32)
    rid = m.upsert("web", "u", "t", "s", "c", v)
    assert rid is not None
    assert m.is_duplicate(v, threshold=0.5) is True
