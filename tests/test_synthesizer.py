from src.agent.synthesizer import synthesize
from src.agent.models import Source

def test_synthesize_formats(monkeypatch):
    def fake_chat(**kwargs):
        class Resp:
            def __init__(self):
                self.choices = [type("C", (), {"message": type("M", (), {"content": "# Title\n\n- A\n\n**Key Takeaways**\n- K1\n\n[1] Ref"})()})]
                self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()
        return Resp()
    
    monkeypatch.setattr("src.agent.synthesizer.client.chat.completions.create", fake_chat)

    sources = [Source(id=1, kind="web", title="T", url="U", snippet="S", content="C")]
    res = synthesize("p", sources, run_id="r1")
    assert "Key Takeaways" in res.markdown
    assert res.tokens_input == 10
