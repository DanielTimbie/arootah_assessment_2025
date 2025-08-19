"""arxiv paper search tool."""
from __future__ import annotations

from dataclasses import dataclass

import arxiv


@dataclass
class ArxivResult:
    """Result from arxiv search."""

    title: str
    url: str
    snippet: str


class ArxivTool:
    """search arxiv for academic papers."""

    def search(self, query: str, k: int = 5) -> list[ArxivResult]:
        """search arxiv papers by query."""
        search = arxiv.Search(
            query=query, max_results=k, sort_by=arxiv.SortCriterion.SubmittedDate
        )
        summary_max_length = 300
        return [
            ArxivResult(
                title=result.title,
                url=result.entry_id,
                snippet=(
                    result.summary[:summary_max_length].replace("\n", " ")
                    + ("..." if len(result.summary) > summary_max_length else "")
                ),
            )
            for result in search.results()
        ]
