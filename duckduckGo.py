#!/usr/bin/env python3
"""
duckduckGo.py: Direct execution of DuckDuckGo search queries.

Demonstrates two ways to query DuckDuckGo directly without agent_toolkits:
  1. Pure Python (zero-LangChain) using the underlying `ddgs` / `duckduckgo_search` library.
     (Recommended for PyRete rule actions where a tool is just a clean Python callable).
  2. Direct LangChain tool import using `DuckDuckGoSearchResults` from `langchain_community.tools`.
"""

import sys
import re
from typing import Any, Dict, List


def search_with_ddgs(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    """
    Direct, lightweight search using the `ddgs` / `duckduckgo_search` library.
    No LangChain dependency required. Returns structured dicts (title, href, body).
    Ideal for PyRete rule productions and facts.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def search_with_langchain_tool(query: str, max_results: int = 6) -> str:
    """
    Direct search using LangChain's DuckDuckGoSearchResults tool
    imported directly from `langchain_community.tools` (no agent_toolkits).
    Configured with max_results to return enough search coverage.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    except ImportError:
        from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
        from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return search_tool.run(query)


def extract_potential_prices(text: str) -> List[str]:
    """Helper regex to spot USD price mentions like $1,999 or $1,599."""
    return re.findall(r"\$[\d,]+(?:\.\d{2})?", text)


def main():
    target_query = "current price of entry-level MacBook Pro in USD"

    print("=" * 80)
    print("🦆 DuckDuckGo Direct Search Runner")
    print(f"🎯 Target Query: \"{target_query}\"")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Approach 1: Pure Python DDGS (Ideal for PyRete facts and rule actions)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Method 1: Direct Python Library (`ddgs` / `duckduckgo_search`)")
    print("No LangChain overhead. Structured dictionaries with max_results=6.")
    print("-" * 80)

    try:
        raw_results = search_with_ddgs(target_query, max_results=6)
        if raw_results:
            for idx, item in enumerate(raw_results, start=1):
                title = item.get("title", "")
                url = item.get("href", "")
                snippet = item.get("body", "")
                prices = extract_potential_prices(snippet)
                price_badge = f" [Prices found: {', '.join(prices)}]" if prices else ""

                print(f"\n[{idx}] {title}{price_badge}")
                print(f"    URL    : {url}")
                print(f"    Snippet: {snippet}")
        else:
            print("No results returned.")
    except Exception as e:
        print(f"Method 1 encountered an error: {e}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Approach 2: Direct LangChain Tool (langchain_community.tools.DuckDuckGoSearchResults)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Method 2: Direct LangChain Tool (`DuckDuckGoSearchResults`)")
    print("Full un-truncated output string with max_results=6.")
    print("-" * 80)

    try:
        tool_output = search_with_langchain_tool(target_query, max_results=6)
        print("\nFull Tool Output String:")
        print(tool_output)

        overall_prices = extract_potential_prices(tool_output)
        if overall_prices:
            print(f"\n💵 Detected Prices in Output: {set(overall_prices)}")
    except Exception as e:
        print(f"Method 2 encountered an error: {e}", file=sys.stderr)

    print("\n" + "=" * 80)
    print("✅ Search execution complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
