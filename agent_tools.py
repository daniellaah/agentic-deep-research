"""Research tool definitions and functions."""

import xml.etree.ElementTree as ET

import requests
from tavily import TavilyClient

MAX_SEARCH_RESULTS = 5
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}

SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "A focused search query."},
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_SEARCH_RESULTS,
        },
    },
    "required": ["query", "max_results"],
    "additionalProperties": False,
}

RESEARCH_TOOLS = [
    {
        "type": "function",
        "name": "tavily_search_tool",
        "description": "Search the web and return titles, snippets, and URLs.",
        "strict": True,
        "parameters": SEARCH_PARAMETERS,
    },
    {
        "type": "function",
        "name": "arxiv_search_tool",
        "description": "Search arXiv and return paper metadata and summaries.",
        "strict": True,
        "parameters": SEARCH_PARAMETERS,
    },
]


def tavily_search_tool(query, max_results, api_key):
    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
    )
    return [
        {
            "title": result.get("title", ""),
            "content": result.get("content", ""),
            "url": result.get("url", ""),
        }
        for result in response.get("results", [])[:max_results]
    ]


def atom_text(entry, path):
    element = entry.find(path, ATOM)
    return " ".join(element.text.split()) if element is not None and element.text else ""


def arxiv_search_tool(query, max_results):
    response = requests.get(
        ARXIV_API_URL,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        },
        headers={
            "User-Agent": (
                "agentic-deep-research/0.5.0 "
                "(+https://github.com/daniellaah/agentic-deep-research)"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    results = []
    for entry in root.findall("atom:entry", ATOM):
        pdf_url = next(
            (
                link.get("href")
                for link in entry.findall("atom:link", ATOM)
                if link.get("title") == "pdf"
            ),
            None,
        )
        results.append(
            {
                "title": atom_text(entry, "atom:title"),
                "authors": [
                    atom_text(author, "atom:name")
                    for author in entry.findall("atom:author", ATOM)
                ],
                "published": atom_text(entry, "atom:published")[:10],
                "url": atom_text(entry, "atom:id"),
                "summary": atom_text(entry, "atom:summary"),
                "pdf_url": pdf_url,
            }
        )
    return results


def execute_tool(name, arguments, tavily_api_key):
    query = arguments.get("query")
    max_results = arguments.get("max_results")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ValueError("max_results must be an integer.")
    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}.")

    if name == "tavily_search_tool":
        return tavily_search_tool(query.strip(), max_results, tavily_api_key)
    if name == "arxiv_search_tool":
        return arxiv_search_tool(query.strip(), max_results)
    raise ValueError(f"Unknown tool: {name}.")
