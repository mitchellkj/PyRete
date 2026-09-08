#!/usr/bin/env python3
"""
react_pyrete_lamma.py: Container-ready ReAct coordinator using PyRete rules & Ollama.

Swaps Google Gemini with a local, open-source LLM via Ollama (e.g. Llama 3.2 1B/3B, Qwen 2.5).
Designed for low-footprint deployment on Hugging Face Spaces or container stacks:
  - Zero proprietary API dependencies (fully self-hosted).
  - Native Ollama HTTP REST interface with fallback to langchain_ollama/langchain_community.
  - Configurable via OLLAMA_HOST and OLLAMA_MODEL environment variables.
  - Uses stop-token truncation ('Observation:') essential for small parameter models.
"""

import os
import re
import sys
import json
import logging
import urllib.request
import urllib.error
import warnings
from typing import Optional, List, Dict, Any

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from py_rete.common import V
from py_rete.fact import Fact
from py_rete.network import ReteNetwork
from py_rete.production import Production

from tool import Tool, DuckDuckGoSearcher, Calculator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Lightweight Ollama Client Wrapper
# =============================================================================

class OllamaResponse:
    """Standardized response object matching LangChain's invoke interface."""

    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f"OllamaResponse(content={self.content[:60]!r}...)"


class OllamaLLM:
    """
    Lightweight client for Ollama's HTTP API.
    Can be configured via:
      - OLLAMA_HOST: URL of the Ollama server (default: http://localhost:11434)
      - OLLAMA_MODEL: Model tag to use (default: llama3.2:3b or llama3.2:1b for minimal footprint)
    """

    def __init__(
            self,
            model: Optional[str] = None,
            base_url: Optional[str] = None,
            temperature: float = 0.0,
            stop: Optional[List[str]] = None,
    ):
        self.base_url = (
                base_url
                or os.environ.get("OLLAMA_HOST")
                or "http://localhost:11434"
        ).rstrip("/")
        # Recommended minimal models for Hugging Face Spaces: llama3.2:1b, llama3.2:3b, qwen2.5:1.5b
        self.model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2:3b"
        self.temperature = temperature
        self.stop = stop or ["\nObservation:", "Observation:"]

        logger.info(
            f"Initialized OllamaLLM using model '{self.model}' at '{self.base_url}'"
        )

    def invoke(self, prompt: str) -> OllamaResponse:
        """Invokes Ollama generate endpoint and returns an OllamaResponse."""
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "stop": self.stop,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result_raw = response.read().decode("utf-8")
                result_json = json.loads(result_raw)
                content = result_json.get("response", "")
                return OllamaResponse(content=content)
        except urllib.error.URLError as e:
            msg = (
                f"Failed to connect to Ollama at '{self.base_url}': {e}.\n"
                f"Make sure Ollama is running (`ollama serve`) and model '{self.model}' is pulled "
                f"(`ollama pull {self.model}`)."
            )
            logger.error(msg)
            raise ConnectionError(msg) from e


# =============================================================================
# 2. Fact Hierarchy for ReAct Working Memory
# =============================================================================

class BaseReActFact(Fact):
    """Base fact ensuring class kind is preserved in working memory."""

    def __init__(self, **kwargs):
        kwargs.setdefault("kind", self.__class__.__name__)
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(**clean_kwargs)


class Goal(BaseReActFact):
    """Goal fact driving forward-chaining activation ('Reason', 'Act', 'Finished')."""

    def __init__(self, goal_type: Optional[str] = None, **kwargs):
        if goal_type is not None:
            kwargs["goal_type"] = goal_type
        super().__init__(**kwargs)


class Question(BaseReActFact):
    """Fact wrapping the user's input question."""

    def __init__(self, text: Optional[str] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        super().__init__(**kwargs)


class Thought(BaseReActFact):
    """Fact wrapping an agent's reasoning step."""

    def __init__(self, text: Optional[str] = None, step: Optional[int] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        if step is not None:
            kwargs["step"] = step
        super().__init__(**kwargs)


class Action(BaseReActFact):
    """Fact wrapping the selected tool name."""

    def __init__(self, name: Optional[str] = None, step: Optional[int] = None, **kwargs):
        if name is not None:
            kwargs["name"] = name
        if step is not None:
            kwargs["step"] = step
        super().__init__(**kwargs)


class ActionInput(BaseReActFact):
    """Fact wrapping the tool input parameter."""

    def __init__(self, text: Optional[str] = None, step: Optional[int] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        if step is not None:
            kwargs["step"] = step
        super().__init__(**kwargs)


class Observation(BaseReActFact):
    """Fact wrapping the tool execution result."""

    def __init__(self, text: Optional[str] = None, step: Optional[int] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        if step is not None:
            kwargs["step"] = step
        super().__init__(**kwargs)


class FinalAnswer(BaseReActFact):
    """Fact wrapping the agent's final answer."""

    def __init__(self, text: Optional[str] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        super().__init__(**kwargs)


class LLMPrediction(BaseReActFact):
    """Fact holding the raw text response from the LLM during Reason phase."""

    def __init__(self, text: Optional[str] = None, step: Optional[int] = None, **kwargs):
        if text is not None:
            kwargs["text"] = text
        if step is not None:
            kwargs["step"] = step
        super().__init__(**kwargs)


class StepTracker(BaseReActFact):
    """Fact tracking current loop step and iteration limits."""

    def __init__(self, current_step: Optional[int] = None, max_steps: Optional[int] = None, **kwargs):
        if current_step is not None:
            kwargs["current_step"] = current_step
        if max_steps is not None:
            kwargs["max_steps"] = max_steps
        super().__init__(**kwargs)


# =============================================================================
# 3. ReAct Prompt Template & Parsing Helpers
# =============================================================================

REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:{scratchpad}"""


def parse_react_response(text: str) -> Dict[str, Any]:
    """
    Parses LLM response text into thought, action, action_input, or final_answer.
    Robust against quirks common in smaller local models.
    """
    cleaned = text.strip()

    # 1. Check for Final Answer
    if "Final Answer:" in cleaned:
        parts = cleaned.split("Final Answer:", 1)
        thought_part = parts[0].replace("Thought:", "").strip()
        final_answer = parts[1].strip()
        return {
            "type": "final_answer",
            "thought": thought_part,
            "final_answer": final_answer,
        }

    # 2. Extract Action & Action Input
    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", cleaned, re.IGNORECASE)
    action = action_match.group(1).strip("[]'\"` ") if action_match else ""

    if "Action Input:" in cleaned:
        action_input_raw = cleaned.split("Action Input:", 1)[1]
        if "\nObservation:" in action_input_raw:
            action_input_raw = action_input_raw.split("\nObservation:", 1)[0]
        elif "Observation:" in action_input_raw:
            action_input_raw = action_input_raw.split("Observation:", 1)[0]
        action_input = action_input_raw.strip().strip("\"'`")
    else:
        action_input = ""

    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", cleaned, re.IGNORECASE | re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    if not thought and not cleaned.lower().startswith("action:"):
        thought = cleaned.split("Action:")[0].strip()

    return {
        "type": "action",
        "thought": thought,
        "action": action,
        "action_input": action_input,
    }


# =============================================================================
# 4. PyRete ReAct Coordinator Class
# =============================================================================

class PyReteReActCoordinator:
    """
    Coordinator managing the forward-chaining ReAct loop via PyRete rules.
    """

    def __init__(
            self,
            llm: Any,
            tools: List[Tool],
            max_iterations: int = 8,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.net = ReteNetwork()

        # Register tools directly into working memory as facts
        for tool in tools:
            self.net.add_fact(tool)

        self._register_productions()

    def _get_fact_by_kind(self, kind: str) -> Optional[Fact]:
        for f in self.net.facts.values():
            if f.get("kind") == kind:
                return f
        return None

    def _get_all_facts_by_kind(self, kind: str) -> List[Fact]:
        return [f for f in self.net.facts.values() if f.get("kind") == kind]

    def _build_scratchpad(self, up_to_step: int) -> str:
        """Assembles previous Thought/Action/Action Input/Observation facts into text."""
        scratchpad_parts = []
        for s in range(1, up_to_step):
            thought = next((f for f in self._get_all_facts_by_kind("Thought") if f.get("step") == s), None)
            action = next((f for f in self._get_all_facts_by_kind("Action") if f.get("step") == s), None)
            action_input = next((f for f in self._get_all_facts_by_kind("ActionInput") if f.get("step") == s), None)
            observation = next((f for f in self._get_all_facts_by_kind("Observation") if f.get("step") == s), None)

            th_text = f" {thought['text']}" if thought and thought.get("text") else ""
            act_text = action["name"] if action else ""
            inp_text = action_input["text"] if action_input else ""
            obs_text = observation["text"] if observation else ""

            step_str = f"{th_text}\nAction: {act_text}\nAction Input: {inp_text}\nObservation: {obs_text}\nThought:"
            scratchpad_parts.append(step_str)

        return "".join(scratchpad_parts)

    def _register_productions(self):
        """Define and attach the Reason and Act productions."""

        # ---------------------------------------------------------------------
        # Rule 1: ReasonRule (Goal == "Reason")
        # ---------------------------------------------------------------------
        @Production((V("g") << Goal(goal_type="Reason")))
        def reason_production(net, g):
            tracker = self._get_fact_by_kind("StepTracker")
            question_fact = self._get_fact_by_kind("Question")
            current_step = tracker["current_step"] if tracker else 1
            max_steps = tracker["max_steps"] if tracker else self.max_iterations

            print(f"\n🧠 [PyRete Reason Phase] Step {current_step} / {max_steps}")

            # Termination safety
            if current_step > max_steps:
                print("⚠️ Max iterations reached without final answer.")
                net.remove_fact(g)
                net.add_fact(Goal(goal_type="Finished"))
                return

            # Format tools
            tools_desc = "\n".join(f"{t.name}: {t.description}" for t in self.tools.values())
            tool_names = ", ".join(self.tools.keys())

            # Build scratchpad from working memory
            scratchpad = self._build_scratchpad(up_to_step=current_step)

            # Assemble prompt
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                tool_names=tool_names,
                question=question_fact["text"] if question_fact else "",
                scratchpad=scratchpad,
            )

            print(f"  • Invoking Ollama ({getattr(self.llm, 'model', 'local')})...")
            response = self.llm.invoke(prompt)
            pred_text = getattr(response, "content", str(response))

            print(f"  • Ollama Prediction:\n{pred_text.strip()}")

            # Forward chain: transition to Act goal
            net.remove_fact(g)
            net.add_fact(LLMPrediction(text=pred_text, step=current_step))
            net.add_fact(Goal(goal_type="Act"))

        # ---------------------------------------------------------------------
        # Rule 2: ActRule (Goal == "Act")
        # ---------------------------------------------------------------------
        @Production(
            (V("g") << Goal(goal_type="Act"))
            & (V("pred") << LLMPrediction())
        )
        def act_production(net, g, pred):
            tracker = self._get_fact_by_kind("StepTracker")
            current_step = tracker["current_step"] if tracker else 1

            print(f"\n⚡ [PyRete Act Phase] Step {current_step}")
            raw_text = pred.get("text", "")

            # Consume prediction and goal facts
            net.remove_fact(g)
            net.remove_fact(pred)

            parsed = parse_react_response(raw_text)

            # Final Answer branch
            if parsed["type"] == "final_answer":
                final_answer = parsed["final_answer"]
                thought_text = parsed["thought"]
                print(f"🎯 [Final Answer Detected]:\n{final_answer}")

                if thought_text:
                    net.add_fact(Thought(text=thought_text, step=current_step))
                net.add_fact(FinalAnswer(text=final_answer))
                net.add_fact(Goal(goal_type="Finished"))
                return

            # Action branch
            thought_text = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", "")

            print(f"  • Thought      : {thought_text}")
            print(f"  • Action       : {action_name}")
            print(f"  • Action Input : {action_input}")

            if thought_text:
                net.add_fact(Thought(text=thought_text, step=current_step))
            net.add_fact(Action(name=action_name, step=current_step))
            net.add_fact(ActionInput(text=action_input, step=current_step))

            # Dispatch tool execution
            selected_tool = self.tools.get(action_name)
            if selected_tool:
                print(f"  • Dispatching Tool '{action_name}'...")
                observation_str = selected_tool.run(action_input)
            else:
                available = ", ".join(self.tools.keys())
                observation_str = f"Error: Tool '{action_name}' not found. Available tools: [{available}]."

            print(f"  • Observation  :\n{observation_str[:300]}" + ("..." if len(observation_str) > 300 else ""))
            net.add_fact(Observation(text=observation_str, step=current_step))

            # Advance step tracker
            if tracker:
                tracker["current_step"] += 1
                net.update_fact(tracker)

            # Cycle back to Reason
            net.add_fact(Goal(goal_type="Reason"))

        self.net.add_production(reason_production)
        self.net.add_production(act_production)

    def run(self, question: str) -> Optional[str]:
        """Runs the PyRete coordinator until Goal == 'Finished' or limit reached."""
        print("=" * 80)
        print("🦙 PyRete + Ollama ReAct Coordinator (Container-Ready)")
        print(f"❓ Question: {question}")
        print("=" * 80)

        self.net.add_fact(Question(text=question))
        self.net.add_fact(StepTracker(current_step=1, max_steps=self.max_iterations))
        self.net.add_fact(Goal(goal_type="Reason"))

        max_firings = (self.max_iterations * 2) + 2
        self.net.run(max_firings)

        final_fact = self._get_fact_by_kind("FinalAnswer")
        final_answer = final_fact["text"] if final_fact else None

        print("\n" + "=" * 80)
        print("📋 PyRete Working Memory Fact Audit:")
        print("=" * 80)
        for f_id, fact in self.net.facts.items():
            kind = fact.get("kind", type(fact).__name__)
            if kind in ("Thought", "Action", "ActionInput", "Observation", "FinalAnswer"):
                print(f"  • [{fact.get('step', '-')}] {kind:<12}: {repr(fact.get('text') or fact.get('name'))[:100]}")

        print("\n" + "=" * 80)
        print("🏁 Execution Complete")
        print("=" * 80)
        if final_answer:
            print(f"Final Answer: {final_answer}\n")
        else:
            print("Finished without explicit FinalAnswer fact.")

        return final_answer


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Configure Ollama parameters via environment (defaults to lightweight llama3.2:3b)
    # For lowest RAM consumption on Hugging Face free tier CPU, set OLLAMA_MODEL="llama3.2:1b"
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    print(f"Initializing Ollama with model: '{model_name}' at '{ollama_host}'")
    ollama_llm = OllamaLLM(
        model=model_name,
        base_url=ollama_host,
        temperature=0.0,
    )

    # Initialize concrete tools (no LangChain wrappers)
    tools = [
        DuckDuckGoSearcher(
            name="duckduck",
            description="A web search engine. Use this to search the web for retail purchase prices, starting MSRP, and specs.",
            max_results=5,
        ),
        Calculator(name="Calculator"),
    ]

    coordinator = PyReteReActCoordinator(
        llm=ollama_llm,
        tools=tools,
        max_iterations=8,
    )

    sample_query = (
        "What is the total retail purchase price (MSRP) of a new entry-level MacBook Pro in USD? "
        "Do not use monthly financing. "
        "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    )
    sample_query_precise = (
        "What is the total retail purchase price (MSRP) of a new entry-level MacBook Pro in USD?"
        "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
        "You must answer BOTH question:"
        "1. The USD MSRP (ignore financing / monthly prices)."
        "2. The equivalent in EUR using the given exchange rate."
        "Always use the Calculator tool for the currency conversion. Do not do the math yourself."
        "Only emit 'Final Answer' after you have both numbers."
    )

    s = """First find the current USD MSRP of a new entry-level MacBook Pro (no financing). 
    Then convert that exact USD amount to EUR using the rate 0.85 EUR = 1 USD. 
    Use the Calculator for the conversion. Report both numbers.
    """

    coordinator.run(s)


if __name__ == "__main__":
    main()
