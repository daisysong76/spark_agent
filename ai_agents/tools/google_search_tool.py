# ai_agent/tools/google_search_tool.py
import logging
import requests
from ai_agent.config.settings import settings

logger = logging.getLogger(__name__)

def google_search(query: str) -> str:
    """
    Uses Google Custom Search API to retrieve search results.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": settings.GOOGLE_API_KEY,
        "cx": settings.GOOGLE_CX,
        "q": query
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        results_str = ""
        for item in items:
            title = item.get("title")
            link = item.get("link")
            results_str += f"{title} => {link}\n"
        logger.info(f"Google Search success: {query}")
        return results_str if results_str else "No results found."
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed Google Search: {e}", exc_info=True)
        raise RuntimeError(f"Google Search request failed: {str(e)}")
