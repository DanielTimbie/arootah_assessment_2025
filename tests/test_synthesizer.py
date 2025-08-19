"""Test synthesizer functionality."""
from src.agent.models import Source
from src.agent.synthesizer import synthesize


def test_synthesize_formats(monkeypatch: object):
    """Test synthesizer output formatting."""
    def fake_chat(*_args: object, **_kwargs: object) -> object:
        class Resp:
            """Mock OpenAI response object."""

            def __init__(self) -> None:
                content = """{
                    "title": "Test Brief",
                    "outline": {
                        "Section 1": ["Point A", "Point B"]
                    },
                    "key_takeaways": ["Key insight 1", "Key insight 2"],
                    "executive_summary": "Test summary with details.",
                    "references": ["[1] Test Author, Test Title, Source, 2024, URL"]
                }"""
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
    assert "Test Brief" in res.markdown
    assert res.tokens_input == 10
