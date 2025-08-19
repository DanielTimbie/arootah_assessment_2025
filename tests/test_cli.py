"""test cli functionality."""
from typer.testing import CliRunner

from src.agent.cli import app


def test_cli_help():
    """test cli help command."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
