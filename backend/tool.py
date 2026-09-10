#!/usr/bin/env python3
"""
backend/tool.py: Concrete Tool hierarchy for the PyRete ReAct Coordinator.
"""

import re
import math
from typing import Optional, Any

from py_rete.fact import Fact


class Tool(Fact):
    """
    Base class representing an agent tool as both a PyRete Fact and an executable callable.
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
        raise NotImplementedError("Subclasses must implement run()")

    def __call__(self, input_text: str) -> str:
        return self.run(input_text)


class DuckDuckGoSearcher(Tool):
    """
    DuckDuckGo search tool using direct ddgs/duckduckgo_search without LangChain wrappers.
    """

    def __init__(
        self,
        name: str = "duckduck",
        description: Optional[str] = None,
        max_results: int = 6,
        **kwargs,
    ):
        desc = description or (
            "A web search engine. Use this to search the web for retail purchase prices, "
            "starting MSRP, and hardware specs."
        )
        super().__init__(name=name, description=desc, max_results=max_results, **kwargs)
        self.max_results = max_results

    def run(self, query: str) -> str:
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
                snippet = (r.get("body") or "")[:250]
                formatted_parts.append(f"snippet: {snippet}, title: {title}, link: {link}")

            return "\n".join(formatted_parts)
        except Exception as e:
            return f"Error executing DuckDuckGo search: {e}"


class Calculator(Tool):
    """
    Calculator tool for mathematical operations and currency conversions.
    Useful for arithmetic and currency conversion. Input should be a mathematical expression, e.g. '1599 * 0.85'.
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

    def _sanitize_expression(self, expr: str) -> str:
        cleaned = expr.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip("\"'` ")
        cleaned = re.sub(r"[$€£¥₹]", "", cleaned)
        cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
        cleaned = re.sub(r"\b(USD|EUR|GBP|JPY|CAD|AUD|CHF|INR|dollars|euros|yen|pounds)\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def run(self, expression: str) -> str:
        cleaned = self._sanitize_expression(expression)
        try:
            import numexpr
            local_dict = {"pi": math.pi, "e": math.e}
            val = numexpr.evaluate(cleaned, local_dict=local_dict)
            float_val = float(val)
            if float_val.is_integer():
                return str(int(float_val))
            return f"{float_val:.2f}"
        except Exception:
            formula_match = re.search(r"[\d.\s+\-*/()^]+", cleaned)
            if formula_match:
                try:
                    import numexpr
                    val = numexpr.evaluate(formula_match.group(0).strip(), local_dict={"pi": math.pi, "e": math.e})
                    return f"{float(val):.2f}"
                except Exception:
                    pass
            return f"Error: Could not parse math expression '{expression}'."
