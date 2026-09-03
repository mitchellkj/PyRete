#!/usr/bin/env python3
"""
react_pyrete.py: ReAct agent coordinator powered by PyRete forward-chaining rules.

Replaces LangChain's AgentExecutor with a goal-driven production rule network:
  - Working memory contains first-class Fact objects for:
      • Goal (alternating "Reason" and "Act" until "Finished")
      • Question, Thought, Action, ActionInput, Observation, FinalAnswer
      • Tools registered directly as facts
  - The "Reason" rule assembles the ReAct prompt from working memory and queries Gemini.
  - The "Act" rule parses the LLM output, executes the selected Tool, asserts the
    Observation fact, and transitions the goal back to "Reason".
"""

import os
import re
import sys
import logging
import warnings
from typing import Optional, List, Dict, Any

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from py_rete.common import V
from py_rete.conditions import Filter
from py_rete.fact import Fact
from py_rete.network import ReteNetwork
from py_rete.production import Production

from langchain_google_genai import ChatGoogleGenerativeAI
from tool import Tool, DuckDuckGoSearcher, Calculator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# 1. Fact Hierarchy for ReAct Working Memory
# =============================================================================

class BaseReActFact(Fact):
    """Base fact ensuring class kind is preserved in working memory."""
    def __init__(self, **kwargs):
        kwargs.setdefault("kind", self.__class__.__name__)
        super().__init__(**kwargs)


class Goal(BaseReActFact):
    """Goal fact driving forward-chaining activation ('Reason', 'Act', 'Finished')."""
    def __init__(self, goal_type: str = "Reason", **kwargs):
        super().__init__(goal_type=goal_type, **kwargs)


class Question(BaseReActFact):
    """Fact wrapping the user's input question."""
    def __init__(self, text: str = "", **kwargs):
        super().__init__(text=text, **kwargs)


class Thought(BaseReActFact):
    """Fact wrapping an agent's reasoning step."""
    def __init__(self, text: str = "", step: int = 1, **kwargs):
        super().__init__(text=text, step=step, **kwargs)


class Action(BaseReActFact):
    """Fact wrapping the selected tool name."""
    def __init__(self, name: str = "", step: int = 1, **kwargs):
        super().__init__(name=name, step=step, **kwargs)


class ActionInput(BaseReActFact):
    """Fact wrapping the tool input parameter."""
    def __init__(self, text: str = "", step: int = 1, **kwargs):
        super().__init__(text=text, step=step, **kwargs)


class Observation(BaseReActFact):
    """Fact wrapping the tool execution result."""
    def __init__(self, text: str = "", step: int = 1, **kwargs):
        super().__init__(text=text, step=step, **kwargs)


class FinalAnswer(BaseReActFact):
    """Fact wrapping the agent's final answer."""
    def __init__(self, text: str = "", **kwargs):
        super().__init__(text=text, **kwargs)


class LLMPrediction(BaseReActFact):
    """Fact holding the raw text response from the LLM during Reason phase."""
    def __init__(self, text: str = "", step: int = 1, **kwargs):
        super().__init__(text=text, step=step, **kwargs)


class StepTracker(BaseReActFact):
    """Fact tracking current loop step and iteration limits."""
    def __init__(self, current_step: int = 1, max_steps: int = 10, **kwargs):
        super().__init__(current_step=current_step, max_steps=max_steps, **kwargs)


# =============================================================================
# 2. ReAct Prompt Template & Parsing Helpers
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

    # 2. Extract Thought, Action, Action Input
    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", cleaned, re.IGNORECASE)
    action_input_match = re.search(r"Action Input:\s*(.*)", cleaned, re.IGNORECASE | re.DOTALL)
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", cleaned, re.IGNORECASE | re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    if not thought and not cleaned.lower().startswith("action:"):
        thought = cleaned.split("Action:")[0].strip()

    action = action_match.group(1).strip() if action_match else ""
    action_input = action_input_match.group(1).strip() if action_input_match else ""

    # Clean action name (remove trailing punctuation or brackets if hallucinated)
    action = action.strip("[]'\"` ")

    return {
        "type": "action",
        "thought": thought,
        "action": action,
        "action_input": action_input,
    }


def get_gemini_api_key(
    secret_id: str = "gemini-api-key",
    version: str = "latest",
    project_id: str = "veytel-cloud-store",
) -> str:
    """Retrieve Gemini API key from environment or Google Cloud Secret Manager."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key.strip()

    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8").strip()
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        return api_key
    except Exception as e:
        raise ValueError(
            f"GEMINI_API_KEY not found in environment and Secret Manager lookup failed: {e}"
        )


# =============================================================================
# 3. PyRete ReAct Coordinator Class
# =============================================================================

class PyReteReActCoordinator:
    """
    Coordinator managing the forward-chaining ReAct loop via PyRete rules.
    """

    def __init__(
        self,
        llm: Any,
        tools: List[Tool],
        max_iterations: int = 10,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.net = ReteNetwork()

        # Register tools directly in working memory as facts
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
        @Production(
            (V("g") << Goal(goal_type="Reason"))
        )
        def reason_production(net, g):
            tracker = self._get_fact_by_kind("StepTracker")
            question_fact = self._get_fact_by_kind("Question")
            current_step = tracker["current_step"] if tracker else 1
            max_steps = tracker["max_steps"] if tracker else self.max_iterations

            print(f"\n🧠 [PyRete Reason Phase] Step {current_step} / {max_steps}")

            # Check termination safety
            if current_step > max_steps:
                print("⚠️ Max iterations reached without final answer.")
                net.remove_fact(g)
                net.add_fact(Goal(goal_type="Finished"))
                return

            # Format tools documentation
            tools_desc = "\n".join(f"{t.name}: {t.description}" for t in self.tools.values())
            tool_names = ", ".join(self.tools.keys())

            # Build scratchpad from working memory facts
            scratchpad = self._build_scratchpad(up_to_step=current_step)

            # Assemble ReAct prompt
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                tool_names=tool_names,
                question=question_fact["text"] if question_fact else "",
                scratchpad=scratchpad,
            )

            print(f"  • Invoking Gemini LLM (step {current_step})...")
            response = self.llm.invoke(prompt)
            pred_text = getattr(response, "content", str(response))

            print(f"  • Gemini Raw Prediction:\n{pred_text.strip()}")

            # Forward chain: transition to Act goal and assert LLMPrediction
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

            # Consume prediction fact and goal
            net.remove_fact(g)
            net.remove_fact(pred)

            parsed = parse_react_response(raw_text)

            # Check if LLM reached the Final Answer
            if parsed["type"] == "final_answer":
                final_answer = parsed["final_answer"]
                thought_text = parsed["thought"]
                print(f"🎯 [Final Answer Detected]:\n{final_answer}")

                if thought_text:
                    net.add_fact(Thought(text=thought_text, step=current_step))
                net.add_fact(FinalAnswer(text=final_answer))
                net.add_fact(Goal(goal_type="Finished"))
                return

            # Otherwise, process Action and Action Input
            thought_text = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", "")

            print(f"  • Thought      : {thought_text}")
            print(f"  • Action       : {action_name}")
            print(f"  • Action Input : {action_input}")

            # Assert working memory facts for this step
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

            # Advance step counter
            if tracker:
                tracker["current_step"] += 1
                net.update_fact(tracker)

            # Transition goal back to Reason for the next iteration
            net.add_fact(Goal(goal_type="Reason"))

        # Attach productions to ReteNetwork
        self.net.add_production(reason_production)
        self.net.add_production(act_production)

    def run(self, question: str) -> Optional[str]:
        """Runs the PyRete coordinator until Goal == 'Finished' or limit reached."""
        print("=" * 80)
        print("🚀 PyRete Goal-Driven ReAct Coordinator")
        print(f"❓ Question: {question}")
        print("=" * 80)

        # Initialize working memory facts
        self.net.add_fact(Question(text=question))
        self.net.add_fact(StepTracker(current_step=1, max_steps=self.max_iterations))
        self.net.add_fact(Goal(goal_type="Reason"))

        # Run the Rete forward-chaining cycle
        # With 2 alternating rules per step, max_steps * 2 firings is plenty
        max_firings = (self.max_iterations * 2) + 2
        self.net.run(max_firings)

        # Retrieve final result
        final_fact = self._get_fact_by_kind("FinalAnswer")
        final_answer = final_fact["text"] if final_fact else None

        # Display working memory breakdown
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
    api_key = get_gemini_api_key()
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    # Initialize concrete tools from tool.py (without LangChain wrappers)
    tools = [
        DuckDuckGoSearcher(name="duckduck", max_results=5),
        Calculator(name="Calculator", llm=gemini_llm),
    ]

    coordinator = PyReteReActCoordinator(
        llm=gemini_llm,
        tools=tools,
        max_iterations=8,
    )

    sample_query = (
        "What is the current price of a MacBook Pro in USD? "
        "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    )

    coordinator.run(sample_query)


if __name__ == "__main__":
    main()
