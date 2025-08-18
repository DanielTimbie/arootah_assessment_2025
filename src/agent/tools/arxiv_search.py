import arxiv
from typing import List, Dict

class ArxivTool:
    def search(self, query: str, k: int = 5) -> List[Dict]:
        search = arxiv.Search(query=query, max_results=k, sort_by=arxiv.SortCriterion.SubmittedDate)
        out: List[Dict] = []
        for result in search.results():
            out.append({
                "title": result.title,
                "url": result.entry_id,
                "snippet": result.summary[:300].replace("\n", " ") + ("..." if len(result.summary) > 300 else ""),
            })
        return out