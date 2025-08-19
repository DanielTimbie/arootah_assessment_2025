"""Test synthesizer functionality."""
from src.agent.models import Source
from src.agent.synthesizer import synthesize


def test_synthesize_formats(monkeypatch: object):
    """Test synthesizer output formatting."""
    def fake_chat(*_: object) -> object:
        class Resp:
            def __init__(self) -> None:
                content = "# Title\n\n- A\n\n**Key Takeaways**\n- K1\n\n[1] Ref"
                msg = type("M", (), {"content": content})
                self.choices = [type("C", (), {"message": msg()})()]
                usage_type = type(
                    "U", (), {"prompt_tokens": 10, "completion_tokens": 5}
                )
                self.usage = usage_type()
        return Resp()

    monkeypatch.setattr(
        "src.agent.synthesizer.client.chat.completions.create", fake_chat
    )

    sources = [Source(id=1, kind="web", title="T", url="U", snippet="S", content="C")]
    res = synthesize("p", sources, run_id="r1")
    assert "Key Takeaways" in res.markdown
    assert res.tokens_input == 10
