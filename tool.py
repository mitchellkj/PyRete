#!/usr/bin/env python3
"""
tool.py: Base and concrete Tool hierarchy for PyRete Coordinator.

Implements:
  - Tool(Fact): Base tool class that is simultaneously a PyRete Fact and an executable callable.
  - DuckDuckGoSearcher(Tool): Direct web search tool without LangChain agent_toolkits/wrappers.
  - Calculator(Tool): Mathematical evaluation and currency converter using numexpr and safe math.
"""

import sys
import re
import math
from typing import Optional, List, Dict, Any

from py_rete.fact import Fact


class Tool(Fact):
    """
    Base class representing an agent tool.
    Inherits from PyRete Fact so instances can be registered directly into
    working memory and queried by productions, while also exposing a .run()
    callable interface.
    """

    def __init__(self, name: str, description: str, **kwargs):
        super().__init__(name=name, description=description, **kwargs)

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def description(self) -> str:
        return self["description"]

    def run(self, input_text: str) -> str:
        """Execute the tool with the given input string and return the result string."""
        raise NotImplementedError("Subclasses must implement run()")

    def __call__(self, input_text: str) -> str:
        return self.run(input_text)


class DuckDuckGoSearcher(Tool):
    """
    DuckDuckGo search tool without LangChain agent_toolkits or wrapper overhead.
    Queries the underlying DDGS engine and formats observations for the ReAct loop.
    """

    def __init__(
        self,
        name: str = "duckduck",
        description: Optional[str] = None,
        max_results: int = 5,
        **kwargs,
    ):
        desc = description or (
            "A web search engine. Use this to search the web for general queries, "
            "latest prices, and news."
        )
        super().__init__(name=name, description=desc, max_results=max_results, **kwargs)
        self.max_results = max_results

    def run(self, query: str) -> str:
        """Executes search and formats snippets matching the ReAct format."""
        query = query.strip("\"' ")
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return f"No search results found for '{query}'."

            formatted_parts = []
            for r in results:
                title = r.get("title", "")
                link = r.get("href", "")
                snippet = r.get("body", "")
                formatted_parts.append(f"snippet: {snippet}, title: {title}, link: {link}")

            return "\n".join(formatted_parts)
        except Exception as e:
            return f"Error executing DuckDuckGo search: {e}"


class Calculator(Tool):
    """
    Calculator tool for mathematical operations and currency conversions.
    Emerged from LangChain's llm-math apparatus:
      - Uses numexpr for high-speed, accurate arithmetic without floating point drift.
      - Includes regex sanitization to handle expressions wrapped in quotes or accompanied by currency labels.
      - Can optionally wrap an LLM / LLMMathChain if natural language math parsing is required.
    """

    def __init__(
        self,
        name: str = "Calculator",
        description: Optional[str] = None,
        llm: Optional[Any] = None,
        **kwargs,
    ):
        desc = description or (
            "Useful for when you need to answer questions about math or perform "
            "currency calculations. Input should be a mathematical expression."
        )
        super().__init__(name=name, description=desc, **kwargs)
        self.llm = llm
        self._llm_chain = None
        if self.llm is not None:
            try:
                from langchain.chains.llm_math.base import LLMMathChain
                self._llm_chain = LLMMathChain.from_llm(llm=self.llm)
            except Exception:
                self._llm_chain = None

    def _sanitize_expression(self, expr: str) -> str:
        """Strip markdown fences, quotes, commas in numbers, and currency symbols."""
        cleaned = expr.strip()
        # Remove code fences if LLM produced them
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip("\"'` ")

        # Remove currency symbols ($ , € , £ , ¥) and commas in numbers like 1,999 -> 1999
        cleaned = re.sub(r"[\$€£¥]", "", cleaned)
        cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)

        # Remove trailing words like USD, EUR, etc.
        cleaned = re.sub(r"\b(USD|EUR|GBP|dollars|euros)\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def run(self, expression: str) -> str:
        """Evaluates numerical expression safely using numexpr or fallback."""
        cleaned = self._sanitize_expression(expression)

        try:
            import numexpr
            local_dict = {"pi": math.pi, "e": math.e}
            val = numexpr.evaluate(cleaned, local_dict=local_dict)
            # Format nicely (e.g. float or int)
            float_val = float(val)
            if float_val.is_integer():
                return str(int(float_val))
            return f"{float_val:.4f}".rstrip("0").rstrip(".")
        except Exception:
            # Fallback 1: Extract basic arithmetic formula via regex
            formula_match = re.search(r"[\d\.\s\+\-\*\/\(\)\^]+", cleaned)
            if formula_match:
                try:
                    import numexpr
                    val = numexpr.evaluate(formula_match.group(0).strip(), local_dict={"pi": math.pi, "e": math.e})
                    return str(float(val))
                except Exception:
                    pass

            # Fallback 2: If LLMChain was provided and expression is complex natural language
            if self._llm_chain is not None:
                try:
                    return str(self._llm_chain.run(expression))
                except Exception as e:
                    return f"Calculator error: {e}"

            return f"Error: Could not parse or evaluate math expression '{expression}'."


# =============================================================================
# Direct Standalone Test Runner
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🛠️ Testing PyRete Tool Hierarchy (tool.py)")
    print("=" * 80)

    # 1. Test DuckDuckGoSearcher
    print("\n--- [1] Testing DuckDuckGoSearcher ---")
    searcher = DuckDuckGoSearcher(max_results=3)
    print(f"Tool Name        : {searcher.name}")
    print(f"Tool Description : {searcher.description}")
    sample_query = "current price of entry-level MacBook Pro in USD"
    print(f"Querying         : '{sample_query}'...")
    search_output = searcher.run(sample_query)
    print("Search Result Snippet:")
    print(search_output[:400] + ("..." if len(search_output) > 400 else ""))

    # 2. Test Calculator
    print("\n--- [2] Testing Calculator ---")
    calculator = Calculator()
    print(f"Tool Name        : {calculator.name}")
    print(f"Tool Description : {calculator.description}")

    expressions = [
        "1999 * 0.85",
        "1,999 USD * 0.85 EUR",
        "$2499 * 0.85",
        "(2000 - 1) * 0.85",
    ]
    for expr in expressions:
        result = calculator.run(expr)
        print(f"  • Expr: {expr:<25} => Result: {result}")

    print("\n" + "=" * 80)
    print("✅ tool.py tests completed.")
    print("=" * 80)
