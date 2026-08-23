"""Research tool definitions and functions."""

import xml.etree.ElementTree as ET

import requests
from tavily import TavilyClient

MAX_RESULTS_PER_SEARCH = 3
MAX_RESULT_TEXT_CHARACTERS = 2000
MAX_SOURCE_CONTENT_CHARACTERS = 6000
MAX_SOURCE_READ_QUERY_CHARACTERS = 500
MAX_EXTRACT_CHUNKS = 5
SOURCE_READ_TIMEOUT_SECONDS = 30
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM = {"atom": "http://www.w3.org/2005/Atom"}

SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "A focused search query."},
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_RESULTS_PER_SEARCH,
        },
    },
    "required": ["query", "max_results"],
    "additionalProperties": False,
}

READ_SOURCE_PARAMETERS = {
    "type": "object",
    "properties": {
        "source_id": {
            "type": "string",
            "description": "A source ID returned by this Worker's search results.",
            "minLength": 1,
        },
        "query": {
            "type": "string",
            "description": "A focused query for relevant content within the source.",
            "minLength": 1,
            "maxLength": MAX_SOURCE_READ_QUERY_CHARACTERS,
        },
    },
    "required": ["source_id", "query"],
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
    {
        "type": "function",
        "name": "read_source_tool",
        "description": (
            "Read bounded relevant content from one source ID returned by this "
            "Worker's searches."
        ),
        "strict": True,
        "parameters": READ_SOURCE_PARAMETERS,
    },
]


def tavily_search_tool(query, max_results, api_key):
    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=max_results,
    )
    results = []
    for result in response.get("results", [])[:max_results]:
        content = result.get("content") or ""
        results.append(
            {
                "title": result.get("title", ""),
                "content": content[:MAX_RESULT_TEXT_CHARACTERS],
                "url": result.get("url", ""),
            }
        )
    return results


def get_atom_text(entry, path):
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
                "agentic-deep-research/0.9.0 "
                "(+https://github.com/daniellaah/agentic-deep-research)"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)

    results = []
    for entry in root.findall("atom:entry", ATOM)[:max_results]:
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
                "title": get_atom_text(entry, "atom:title"),
                "authors": [
                    get_atom_text(author, "atom:name")
                    for author in entry.findall("atom:author", ATOM)
                ],
                "published": get_atom_text(entry, "atom:published")[:10],
                "url": get_atom_text(entry, "atom:id"),
                "summary": get_atom_text(entry, "atom:summary")[
                    :MAX_RESULT_TEXT_CHARACTERS
                ],
                "pdf_url": pdf_url,
            }
        )
    return results


def read_source_tool(source_url, query, api_key):
    response = TavilyClient(api_key=api_key).extract(
        urls=source_url,
        query=query,
        chunks_per_source=MAX_EXTRACT_CHUNKS,
        extract_depth="basic",
        format="markdown",
        timeout=SOURCE_READ_TIMEOUT_SECONDS,
    )
    if not isinstance(response, dict):
        raise TypeError("Source extraction returned an invalid response.")

    results = response.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("Source extraction must return exactly one result.")
    result = results[0]
    if not isinstance(result, dict):
        raise TypeError("Source extraction returned an invalid result.")

    extracted_url = result.get("url")
    raw_content = result.get("raw_content")
    if not isinstance(extracted_url, str) or not extracted_url.strip():
        raise ValueError("Source extraction returned no destination URL.")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ValueError("Source extraction returned no content.")

    content = raw_content[:MAX_SOURCE_CONTENT_CHARACTERS]
    return [
        {
            "url": extracted_url.strip(),
            "content": content,
            "content_characters": len(content),
            "truncated": len(raw_content) > len(content),
        }
    ]


def require_query(arguments, *, max_length=None):
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    query = query.strip()
    if max_length is not None and len(query) > max_length:
        raise ValueError(f"query must be at most {max_length} characters.")
    return query


def execute_research_tool(
    name,
    arguments,
    tavily_api_key,
    *,
    source_url=None,
):
    if name == "read_source_tool":
        query = require_query(
            arguments,
            max_length=MAX_SOURCE_READ_QUERY_CHARACTERS,
        )
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError(
                "read_source_tool requires an application-resolved source URL."
            )
        return read_source_tool(source_url, query, tavily_api_key)

    query = require_query(arguments)
    max_results = arguments.get("max_results")
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ValueError("max_results must be an integer.")
    if not 1 <= max_results <= MAX_RESULTS_PER_SEARCH:
        raise ValueError(
            f"max_results must be between 1 and {MAX_RESULTS_PER_SEARCH}."
        )

    if name == "tavily_search_tool":
        return tavily_search_tool(query, max_results, tavily_api_key)
    if name == "arxiv_search_tool":
        return arxiv_search_tool(query, max_results)
    raise ValueError(f"Unknown tool: {name}.")
