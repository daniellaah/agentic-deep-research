"""Tool implementations and Responses API schemas for the learning agent."""

import os
import xml.etree.ElementTree as ET

import requests
from tavily import TavilyClient

__all__ = [
    "arxiv_search_tool",
    "arxiv_tool_def",
    "tavily_search_tool",
    "tavily_tool_def",
]


_ARXIV_SESSION = requests.Session()
_ARXIV_SESSION.headers.update({"User-Agent": "LF-ADP-Agent/1.0 (mailto:your.email@example.com)"})


def _required_text(
    parent: ET.Element,
    path: str,
    namespace: dict[str, str],
) -> str:
    """Return required XML text or fail with a useful parsing error."""
    element = parent.find(path, namespace)
    if element is None or element.text is None:
        raise ValueError(f"Missing required arXiv field: {path}")
    return element.text


def arxiv_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for research papers matching the query."""
    query = query.strip()
    if not query:
        raise ValueError("arXiv query must not be empty.")

    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        response = _ARXIV_SESSION.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return [{"error": str(exc)}]

    try:
        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for entry in root.findall("atom:entry", namespace):
            title = _required_text(entry, "atom:title", namespace).strip()
            authors = [
                _required_text(author, "atom:name", namespace)
                for author in entry.findall("atom:author", namespace)
            ]
            published = _required_text(entry, "atom:published", namespace)[:10]
            url_abstract = _required_text(entry, "atom:id", namespace)
            summary = _required_text(entry, "atom:summary", namespace).strip()

            link_pdf = None
            for link in entry.findall("atom:link", namespace):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            results.append(
                {
                    "title": title,
                    "authors": authors,
                    "published": published,
                    "url": url_abstract,
                    "summary": summary,
                    "link_pdf": link_pdf,
                }
            )

        return results
    except Exception as exc:  # noqa: BLE001 - return parsing failures to the agent
        return [{"error": f"Parsing failed: {exc}"}]


arxiv_tool_def = {
    "type": "function",
    "name": "arxiv_search_tool",
    "description": (
        "Searches arXiv for academic papers using plain semantic keywords. "
        "Use it for papers, methods, experiments, and benchmarks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Plain semantic keywords for academic papers. Do not include URLs, "
                    "site operators, the word arXiv, or web-style date ranges such as "
                    "2020..2026."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def tavily_search_tool(
    query: str, max_results: int = 5, include_images: bool = False
) -> list[dict]:
    """Perform a general-purpose web search using the Tavily API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")

    api_base_url = os.getenv("DLAI_TAVILY_BASE_URL")

    try:
        if api_base_url:
            client = TavilyClient(api_key=api_key, api_base_url=api_base_url)
        else:
            client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images,
        )

        results = [
            {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
            }
            for result in response.get("results", [])
        ]

        if include_images:
            results.extend({"image_url": image_url} for image_url in response.get("images", []))

        return results
    except Exception as exc:  # noqa: BLE001 - return tool failures to the agent
        return [{"error": str(exc)}]


tavily_tool_def = {
    "type": "function",
    "name": "tavily_search_tool",
    "description": "Performs a general-purpose web search using the Tavily API.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords for retrieving information from the web.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
            },
            "include_images": {
                "type": "boolean",
                "description": "Whether to include image results.",
                "default": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}
