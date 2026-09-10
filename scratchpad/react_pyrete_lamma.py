#!/usr/bin/env python3
"""
react_pyrete_lamma.py: Replaced by Gemini-based PyRete ReAct Coordinator with Working Memory Persistence.

Ollama has been removed from the architecture. This module delegates directly to
react_pyrete.py, utilizing Google Gemini API accessibility and persistent working memory
keyed by prompt / user query hash.
"""

from react_pyrete import (
    BaseReActFact,
    Goal,
    Finished,
    Question,
    Thought,
    Action,
    ActionInput,
    Observation,
    FinalAnswer,
    LLMPrediction,
    StepTracker,
    FACT_CLASS_MAP,
    REACT_PROMPT_TEMPLATE,
    parse_react_response,
    get_gemini_api_key,
    create_gemini_llm,
    WorkingMemoryStore,
    PyReteReActCoordinator,
    main,
)

__all__ = [
    "BaseReActFact",
    "Goal",
    "Finished",
    "Question",
    "Thought",
    "Action",
    "ActionInput",
    "Observation",
    "FinalAnswer",
    "LLMPrediction",
    "StepTracker",
    "FACT_CLASS_MAP",
    "REACT_PROMPT_TEMPLATE",
    "parse_react_response",
    "get_gemini_api_key",
    "create_gemini_llm",
    "WorkingMemoryStore",
    "PyReteReActCoordinator",
    "main",
]

if __name__ == "__main__":
    main()
