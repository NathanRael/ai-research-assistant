from langchain_core.tools import BaseTool, tool

from app.tools.web_search_client import SearchResult, WebSearchClient


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."
    blocks = [
        f"{idx}. {result['title']}\n{result['content']}\nSource: {result['source']}"
        for idx, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def create_search_web_tool(client: WebSearchClient) -> BaseTool:
    """Wrap the existing web search client into a LangChain tool."""

    @tool
    def search_web(query: str) -> str:
        """Search the web for up-to-date information relevant to the query.

        Use a single optimized query. Only run additional searches if the first
        result set is clearly insufficient.
        """
        try:
            results = client.search(query)
        except Exception as exc:
            return (
                "The search service is currently unavailable. "
                f"Error: {exc}\n"
                "Please try again in a moment or ask without web search."
            )
        return _format_results(results)

    return search_web
