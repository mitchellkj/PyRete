#!/usr/bin/env python3
"""
react_pyrete.py: ReAct agent coordinator powered by PyRete forward-chaining rules.

Strategy:
  1. Session-Scoped Working Memory: Facts maintain session cookies/identifiers (`session_id`)
     allowing concurrent multi-user queries to live side-by-side in working memory without interference.
  2. Robust Fact Hashability: Enforces that all Fact attributes are hashable, preventing
     py_rete WME hash failures (TypeError: unhashable type: 'list') when Gemini outputs structured content parts.
  3. Goal session-tagging and session-scoped production rules.
  4. Persistent Working Memory Store keyed by prompt / User Query hash (SHA-256).
  5. First checks for cached Working Memory; if present, unmarshals facts and substitutes the
     current session id with the persisted one, returning immediately on Finished fact detection.
  6. Working Memory is cleanly cleared upon Finished resolution.
  7. Checks Gemini API accessibility before inference: if accessible, lets ReAct execute through
     Gemini and persists the deduced facts; otherwise fails gracefully and does NOT persist WM.
"""

import os
import re
import sys
import json
import uuid
import hashlib
import logging
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

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
# 0. Text Extraction Helper for LLM Responses
# =============================================================================

def extract_text_from_response(response: Any) -> str:
    """
    Safely extracts string text from LLM response objects.
    Handles strings, lists of content blocks, text part dicts, and objects with .text/.content.
    Prevents unhashable list/dict objects from entering py_rete Fact WMEs.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]))
                elif "content" in item and item["content"]:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            elif hasattr(item, "text") and item.text:
                parts.append(str(item.text))
            elif hasattr(item, "content") and item.content:
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        return "".join(parts)

    if hasattr(content, "text"):
        return str(content.text)

    return str(content)


# =============================================================================
# 1. Fact Hierarchy for Session-Scoped Working Memory
# =============================================================================

class BaseReActFact(Fact):
    """
    Base fact ensuring class kind, step, and session identifier are preserved
    in working memory as a first-class frequency/dimension.
    Guarantees all values are hashable (converting lists/dicts to strings or tuples)
    to prevent py_rete WME hash failures (TypeError: unhashable type: 'list').
    Preserves py_rete Variable (V) objects when used in production rule patterns.
    """

    def __init__(
            self,
            session_id: Optional[Any] = None,
            step: Optional[Any] = None,
            **kwargs,
    ):
        kwargs.setdefault("kind", self.__class__.__name__)
        if session_id is not None:
            kwargs["session_id"] = session_id if isinstance(session_id, V) else str(session_id)
        if step is not None:
            kwargs["step"] = step if isinstance(step, V) else step

        clean_kwargs = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, V):
                clean_kwargs[k] = v
            elif isinstance(v, list):
                if all(isinstance(item, str) for item in v):
                    clean_kwargs[k] = "\n".join(v)
                else:
                    clean_kwargs[k] = tuple(v)
            elif isinstance(v, dict):
                clean_kwargs[k] = tuple(sorted((str(k_), str(v_)) for k_, v_ in v.items()))
            else:
                clean_kwargs[k] = v

        super().__init__(**clean_kwargs)


class Goal(BaseReActFact):
    """Goal fact driving forward-chaining activation scoped to session_id."""

    def __init__(
            self,
            goal_type: Optional[Any] = None,
            session_id: Optional[Any] = None,
            step: Optional[Any] = None,
            **kwargs,
    ):
        if goal_type is not None:
            kwargs["goal_type"] = goal_type if isinstance(goal_type, V) else str(goal_type)
        super().__init__(session_id=session_id, step=step, **kwargs)


class Finished(BaseReActFact):
    """Fact indicating coordinator execution for a given session has completed."""

    def __init__(
            self,
            session_id: Optional[Any] = None,
            step: Optional[Any] = None,
            **kwargs,
    ):
        super().__init__(session_id=session_id, step=step, **kwargs)


class Question(BaseReActFact):
    """Fact wrapping the user's input question for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            session_id: Optional[Any] = None,
            step: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class Thought(BaseReActFact):
    """Fact wrapping an agent's reasoning step for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class Action(BaseReActFact):
    """Fact wrapping the selected tool name for a session."""

    def __init__(
            self,
            name: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if name is not None:
            kwargs["name"] = name if isinstance(name, V) else str(name)
        super().__init__(session_id=session_id, step=step, **kwargs)


class ActionInput(BaseReActFact):
    """Fact wrapping the tool input parameter for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class Observation(BaseReActFact):
    """Fact wrapping the tool execution result for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class FinalAnswer(BaseReActFact):
    """Fact wrapping the agent's final answer for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class LLMPrediction(BaseReActFact):
    """Fact holding raw LLM text response during Reason phase for a session."""

    def __init__(
            self,
            text: Optional[Any] = None,
            step: Optional[Any] = None,
            session_id: Optional[Any] = None,
            **kwargs,
    ):
        if text is not None:
            kwargs["text"] = text if isinstance(text, V) else extract_text_from_response(text)
        super().__init__(session_id=session_id, step=step, **kwargs)


class StepTracker(BaseReActFact):
    """Fact tracking current loop step and iteration limits for a session."""

    def __init__(
            self,
            current_step: Optional[Any] = None,
            max_steps: Optional[Any] = None,
            session_id: Optional[Any] = None,
            step: Optional[Any] = None,
            **kwargs,
    ):
        if current_step is not None:
            kwargs["current_step"] = current_step if isinstance(current_step, V) else int(current_step)
        if max_steps is not None:
            kwargs["max_steps"] = max_steps if isinstance(max_steps, V) else int(max_steps)
        super().__init__(session_id=session_id, step=step, **kwargs)


FACT_CLASS_MAP = {
    "Goal": Goal,
    "Finished": Finished,
    "Question": Question,
    "Thought": Thought,
    "Action": Action,
    "ActionInput": ActionInput,
    "Observation": Observation,
    "FinalAnswer": FinalAnswer,
    "StepTracker": StepTracker,
    "LLMPrediction": LLMPrediction,
}

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


def parse_react_response(text: Any) -> Dict[str, Any]:
    cleaned = extract_text_from_response(text).strip()

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

    action = action.strip("[]'\"` ")

    return {
        "type": "action",
        "thought": thought,
        "action": action,
        "action_input": action_input,
    }


# =============================================================================
# 3. Gemini API Integration
# =============================================================================

def get_gemini_api_key(
        secret_id: str = "gemini-api-key",
        version: str = "latest",
        project_id: str = "veytel-cloud-store",
) -> Optional[str]:
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
        logger.warning(f"Could not retrieve Gemini API key from Secret Manager: {e}")
        return None


def create_gemini_llm(model: Optional[str] = None, temperature: float = 0.0) -> Any:
    """Creates the Gemini LLM client. Fails gracefully if key or library is missing."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ConnectionError(
            "Gemini API key not found in environment (GEMINI_API_KEY/GOOGLE_API_KEY) "
            "and Secret Manager lookup failed."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


# =============================================================================
# 4. Working Memory Persistence
# =============================================================================

def get_default_wm_dir() -> str:
    """Returns the default directory for persistent working memory."""
    if "WM_STORAGE_DIR" in os.environ:
        return os.environ["WM_STORAGE_DIR"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "wm_storage")


class WorkingMemoryStore:
    """
    Persists working memory facts to disk keyed by the prompt / User Query hash (SHA-256).
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or get_default_wm_dir()
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Unable to create WM storage directory '{self.storage_dir}': {e}")

    @staticmethod
    def compute_query_hash(query: str) -> str:
        """Computes SHA-256 hash for prompt or User Query."""
        normalized = query.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_file_path(self, query_hash: str) -> str:
        return os.path.join(self.storage_dir, f"{query_hash}.json")

    def exists(self, query: str) -> bool:
        """Checks if persisted working memory exists for query."""
        query_hash = self.compute_query_hash(query)
        path = self.get_file_path(query_hash)
        return os.path.exists(path)

    def load(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Loads facts from persisted working memory."""
        query_hash = self.compute_query_hash(query)
        path = self.get_file_path(query_hash)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("facts", [])
        except Exception as e:
            logger.error(f"Failed to load persisted WM from '{path}': {e}")
            return None

    def save(self, query: str, facts: List[Fact]) -> bool:
        """
        Saves episodic facts to persisted working memory.
        Only Gemini-deduced facts are saved.
        """
        query_hash = self.compute_query_hash(query)
        path = self.get_file_path(query_hash)
        serializable_facts = []
        for f in facts:
            kind = f.get("kind")
            if kind in (
                    "Question",
                    "Goal",
                    "Finished",
                    "Thought",
                    "Action",
                    "ActionInput",
                    "Observation",
                    "FinalAnswer",
                    "StepTracker",
            ):
                fact_dict = {k: v for k, v in f.items() if not k.startswith("_")}
                serializable_facts.append(fact_dict)

        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            payload = {
                "query_hash": query_hash,
                "query": query,
                "created_at": datetime.now().isoformat(),
                "facts": serializable_facts,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(
                f"Successfully persisted working memory ({len(serializable_facts)} facts) "
                f"to '{path}' [hash: {query_hash[:10]}...]"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to persist working memory to '{path}': {e}")
            return False


# =============================================================================
# 5. Session-Aware PyRete ReAct Coordinator Class
# =============================================================================

class PyReteReActCoordinator:
    """
    Coordinator managing the forward-chaining ReAct loop via PyRete rules.
    Maintains session integrity by scoping Goals and episodic facts by `session_id`.
    Guarantees all facts in working memory contain hashable attributes.
    """

    def __init__(
            self,
            llm: Optional[Any] = None,
            tools: Optional[List[Tool]] = None,
            max_iterations: int = 10,
            storage_dir: Optional[str] = None,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.wm_store = WorkingMemoryStore(storage_dir=storage_dir)

        default_tools = [
            DuckDuckGoSearcher(name="duckduck", max_results=5),
            Calculator(name="Calculator", llm=self.llm),
        ]
        self.tools = {tool.name: tool for tool in (tools or default_tools)}

        # Long-lived PyRete network with compiled production rules
        self.net = ReteNetwork()
        for tool in self.tools.values():
            self.net.add_fact(tool)

        # Track session-scoped execution errors
        self.session_errors: Dict[str, str] = {}

        # Compile and register productions ONCE
        self._register_productions()

    def check_gemini_accessibility(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies that Gemini API exists and can connect.
        Returns (True, None) if accessible, or (False, error_message).
        Explicitly distinguishes quota exhaustion (429) from connectivity errors.
        """
        if self.llm is None:
            try:
                self.llm = create_gemini_llm()
            except Exception as e:
                return False, f"Gemini API initialization failed: {e}"

        try:
            response = self.llm.invoke("ping")
            # Guard: empty response from a quota-exhausted API may still not raise
            text = extract_text_from_response(response)
            if not text.strip():
                logger.warning("Gemini API returned empty response on connectivity check — possible quota exhaustion.")
                return False, (
                    "Gemini API returned an empty response. This typically indicates quota exhaustion "
                    "(free-tier daily limit reached). Please wait for quota reset or upgrade your plan."
                )
            return True, None
        except Exception as e:
            err_str = str(e)
            if any(kw in err_str for kw in ("429", "ResourceExhausted", "quota", "RESOURCE_EXHAUSTED", "rate limit")):
                msg = (
                    f"Gemini API quota exhausted (429 ResourceExhausted). "
                    f"Your free-tier or daily limit has been reached. "
                    f"Please wait for quota reset or upgrade your plan. Detail: {e}"
                )
                logger.warning(f"Gemini API quota check failed: {msg}")
                return False, msg
            logger.warning(f"Gemini API connectivity verification failed: {e}")
            return False, f"Gemini connection failed: {e}"

    def clear_session_facts(self, session_id: str) -> None:
        """
        Removes all working memory facts associated with a session_id from the Rete network.
        Ensures working memory is cleanly cleared upon Finished resolution.
        """
        to_remove = [
            f for f in list(self.net.facts.values())
            if f.get("session_id") == session_id
        ]
        for f in to_remove:
            try:
                self.net.remove_fact(f)
            except Exception as e:
                logger.warning(f"Error removing fact {f.get('kind')} for session {session_id}: {e}")

    def _is_session_finished(self, session_id: str) -> bool:
        """Checks if the network contains a Finished fact or Finished goal for session_id."""
        for f in self.net.facts.values():
            if f.get("session_id") == session_id:
                if f.get("kind") == "Finished":
                    return True
                if f.get("kind") == "Goal" and f.get("goal_type") == "Finished":
                    return True
        return False

    def _get_fact_by_kind(self, kind: str, session_id: str) -> Optional[Fact]:
        for f in self.net.facts.values():
            if f.get("kind") == kind and f.get("session_id") == session_id:
                return f
        return None

    def _build_scratchpad(self, session_id: str, up_to_step: int) -> str:
        """Assembles previous steps for a specific session_id into ReAct scratchpad."""
        scratchpad_parts = []
        for s in range(1, up_to_step):
            thought = next((f for f in self.net.facts.values() if
                            f.get("kind") == "Thought" and f.get("session_id") == session_id and f.get("step") == s),
                           None)
            action = next((f for f in self.net.facts.values() if
                           f.get("kind") == "Action" and f.get("session_id") == session_id and f.get("step") == s),
                          None)
            action_input = next((f for f in self.net.facts.values() if
                                 f.get("kind") == "ActionInput" and f.get("session_id") == session_id and f.get(
                                     "step") == s), None)
            observation = next((f for f in self.net.facts.values() if
                                f.get("kind") == "Observation" and f.get("session_id") == session_id and f.get(
                                    "step") == s), None)

            th_text = f" {thought['text']}" if thought and thought.get("text") else ""
            act_text = action["name"] if action else ""
            inp_text = action_input["text"] if action_input else ""
            obs_text = observation["text"] if observation else ""

            step_str = f"{th_text}\nAction: {act_text}\nAction Input: {inp_text}\nObservation: {obs_text}\nThought:"
            scratchpad_parts.append(step_str)

        return "".join(scratchpad_parts)

    def _register_productions(self):
        """Define and attach session-scoped Reason and Act productions."""

        # ---------------------------------------------------------------------
        # Rule 1: ReasonRule (session-scoped)
        # ---------------------------------------------------------------------
        @Production(
            (V("g") << Goal(goal_type="Reason", session_id=V("sid")))
            & (V("tracker") << StepTracker(session_id=V("sid")))
            & (V("q") << Question(session_id=V("sid")))
        )
        def reason_production(net, g, tracker, q, sid):
            current_step = tracker.get("current_step", 1)
            max_steps = tracker.get("max_steps", self.max_iterations)

            print(f"\n🧠 [PyRete Reason Phase | Session: {sid[:8]}] Step {current_step} / {max_steps}")

            # Check termination safety
            if current_step > max_steps:
                print("⚠️ Max iterations reached without final answer.")
                net.remove_fact(g)
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            # Format tools documentation
            tools_desc = "\n".join(f"{t.name}: {t.description}" for t in self.tools.values())
            tool_names = ", ".join(self.tools.keys())

            # Build scratchpad for this session
            scratchpad = self._build_scratchpad(session_id=sid, up_to_step=current_step)

            # Assemble ReAct prompt
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                tool_names=tool_names,
                question=q.get("text", ""),
                scratchpad=scratchpad,
            )

            print(f"  • Invoking Gemini LLM (step {current_step})...")
            try:
                if self.llm is None:
                    self.llm = create_gemini_llm()
                response = self.llm.invoke(prompt)
                # Safely extract text string to prevent unhashable list/dict WMEs
                pred_text = extract_text_from_response(response)
            except Exception as e:
                err_str = str(e)
                # Detect quota/rate-limit errors explicitly for a clear failure message
                if any(kw in err_str for kw in ("429", "ResourceExhausted", "quota", "RESOURCE_EXHAUSTED", "rate limit")):
                    logger.error(f"[Session {sid}] Gemini API quota exhausted at step {current_step}: {e}")
                    self.session_errors[sid] = (
                        f"Gemini API quota exhausted (429 ResourceExhausted). "
                        f"Your free-tier or daily limit has been reached. "
                        f"Please wait for quota reset or upgrade your plan. Detail: {e}"
                    )
                else:
                    logger.error(f"[Session {sid}] Gemini API connection error: {e}", exc_info=True)
                    self.session_errors[sid] = f"Gemini API cannot connect: {e}"
                net.remove_fact(g)
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            # Guard: empty response indicates quota exhaustion returning empty content
            if not pred_text.strip():
                logger.error(
                    f"[Session {sid}] Gemini returned empty response at step {current_step}. "
                    "This typically means quota exhaustion or model unavailability."
                )
                self.session_errors[sid] = (
                    "Gemini returned an empty response. This is typically caused by API quota exhaustion "
                    "(free-tier daily limit reached). Please wait for quota reset or check your API plan."
                )
                net.remove_fact(g)
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            print(f"  • Gemini Raw Prediction:\n{pred_text.strip()}")

            net.remove_fact(g)
            net.add_fact(LLMPrediction(session_id=sid, text=pred_text, step=current_step))
            net.add_fact(Goal(session_id=sid, goal_type="Act", step=current_step))

        # ---------------------------------------------------------------------
        # Rule 2: ActRule (session-scoped)
        # ---------------------------------------------------------------------
        @Production(
            (V("g") << Goal(goal_type="Act", session_id=V("sid")))
            & (V("pred") << LLMPrediction(session_id=V("sid")))
            & (V("tracker") << StepTracker(session_id=V("sid")))
        )
        def act_production(net, g, pred, tracker, sid):
            current_step = tracker.get("current_step", 1)
            raw_text = pred.get("text", "")

            # Consume prediction fact and goal
            net.remove_fact(g)
            net.remove_fact(pred)

            parsed = parse_react_response(raw_text)

            # Check if LLM reached the Final Answer
            if parsed["type"] == "final_answer":
                final_answer = parsed["final_answer"]
                thought_text = parsed["thought"]
                print(f"🎯 [Final Answer Detected | Session: {sid[:8]}]:\n{final_answer}")

                if thought_text:
                    net.add_fact(Thought(session_id=sid, text=thought_text, step=current_step))
                net.add_fact(FinalAnswer(session_id=sid, text=final_answer, step=current_step))
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            thought_text = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", "")

            print(f"  • Thought      : {thought_text}")
            print(f"  • Action       : {action_name}")
            print(f"  • Action Input : {action_input}")

            if thought_text:
                net.add_fact(Thought(session_id=sid, text=thought_text, step=current_step))
            net.add_fact(Action(session_id=sid, name=action_name, step=current_step))
            net.add_fact(ActionInput(session_id=sid, text=action_input, step=current_step))

            # Dispatch tool execution with case-insensitive fallback
            selected_tool = self.tools.get(action_name)
            if not selected_tool:
                for t_name, t_inst in self.tools.items():
                    if t_name.lower() == action_name.lower():
                        selected_tool = t_inst
                        break

            if selected_tool:
                print(f"  • Dispatching Tool '{selected_tool.name}'...")
                observation_str = selected_tool.run(action_input)
            else:
                available = ", ".join(self.tools.keys())
                observation_str = f"Error: Tool '{action_name}' not found. Available tools: [{available}]."

            print(f"  • Observation  :\n{observation_str[:300]}" + ("..." if len(observation_str) > 300 else ""))
            net.add_fact(Observation(session_id=sid, text=observation_str, step=current_step))

            tracker["current_step"] = current_step + 1
            net.update_fact(tracker)
            net.add_fact(Goal(session_id=sid, goal_type="Reason", step=tracker["current_step"]))

        self.net.add_production(reason_production)
        self.net.add_production(act_production)

    def _run_until_session_finished(self, session_id: str, max_firings: int) -> None:
        """Executes rule firings prioritizing the specified session until it finishes."""
        firings = 0
        while firings < max_firings:
            if self._is_session_finished(session_id):
                break

            matches = list(self.net.matches)
            if not matches:
                break

            session_matches = [
                m for m in matches
                if m.token.binding.get(V("sid")) == session_id
            ]
            chosen = session_matches[0] if session_matches else matches[0]
            chosen.fire()
            firings += 1

    def _print_fact_audit(self, session_id: str):
        """Displays working memory breakdown for session_id."""
        print("\n" + "=" * 80)
        print(f"📋 PyRete Working Memory Fact Audit [Session: {session_id}]:")
        print("=" * 80)
        for f_id, fact in self.net.facts.items():
            if fact.get("session_id") == session_id:
                kind = fact.get("kind", type(fact).__name__)
                if kind in ("Thought", "Action", "ActionInput", "Observation", "FinalAnswer"):
                    print(
                        f"  • [{fact.get('step', '-')}] {kind:<12}: {repr(fact.get('text') or fact.get('name'))[:100]}")

    def run(self, question: str, session_id: Optional[str] = None) -> Optional[str]:
        """
        Runs the coordinator following the session-scoped and persisted working memory strategy:
          1. Sets or generates session_id.
          2. Clears previous facts for this session_id from working memory.
          3. Checks for persisted working memory (keyed by query hash):
             - If found, unmarshals facts and substitutes session_id.
             - Returns immediately on Finished fact.
             - Clears working memory upon Finished resolution.
          4. If not found, checks Gemini Accessibility:
             - If NOT accessible: fails gracefully and does NOT persist WM.
             - If accessible: runs ReAct loop through Gemini.
             - If successful, persists deduced Working Memory.
             - Clears working memory upon Finished resolution.
        """
        sid = session_id or f"cli-{uuid.uuid4().hex[:8]}"
        query_hash = self.wm_store.compute_query_hash(question)

        # Clear any prior state or leftover facts for this session to ensure a clean slate
        self.clear_session_facts(sid)
        self.session_errors.pop(sid, None)

        print("=" * 80)
        print("🚀 PyRete Goal-Driven ReAct Coordinator")
        print(f"❓ Question   : {question}")
        print(f"🔑 Query Hash : {query_hash}")
        print(f"🆔 Session ID : {sid}")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # 1 & 2. Check Persisted Working Memory
        # ---------------------------------------------------------------------
        if self.wm_store.exists(question):
            print(f"\n📦 [Working Memory Cache Hit] Found persisted WM for hash: {query_hash}")
            print(f"  • Unmarshaling facts and substituting current session_id '{sid}'...")
            persisted_facts = self.wm_store.load(question)
            if persisted_facts:
                for item in persisted_facts:
                    item_copy = dict(item)
                    item_copy["session_id"] = sid
                    kind = item_copy.get("kind")
                    cls = FACT_CLASS_MAP.get(kind, BaseReActFact)
                    kwargs = {k: v for k, v in item_copy.items() if k != "kind"}
                    self.net.add_fact(cls(**kwargs))

                if self._is_session_finished(sid):
                    print("✨ Finished Fact detected in restored Working Memory - immediately returning!")
                    final_fact = self._get_fact_by_kind("FinalAnswer", sid)
                    final_answer = final_fact["text"] if final_fact else None
                    self._print_fact_audit(sid)

                    # Clear working memory upon Finished resolution
                    self.clear_session_facts(sid)

                    print("\n" + "=" * 80)
                    print("🏁 Execution Complete (Cached Working Memory)")
                    print("=" * 80)
                    if final_answer:
                        print(f"Final Answer: {final_answer}\n")
                    return final_answer

        # ---------------------------------------------------------------------
        # 3. Persisted WM does not exist -> Check Gemini Accessibility
        # ---------------------------------------------------------------------
        print("  • Verifying Gemini API accessibility...")
        is_accessible, access_error = self.check_gemini_accessibility()
        if not is_accessible:
            print(f"\n⚠️ [Graceful Failure] Gemini API is unavailable or cannot connect: {access_error}")
            print("  • Working Memory will NOT be persisted.")
            self.clear_session_facts(sid)
            return None

        # ---------------------------------------------------------------------
        # 4. Let the ReAct process execute through Gemini
        # ---------------------------------------------------------------------
        self.net.add_fact(Question(session_id=sid, text=question, step=0))
        self.net.add_fact(StepTracker(session_id=sid, current_step=1, max_steps=self.max_iterations, step=0))
        self.net.add_fact(Goal(session_id=sid, goal_type="Reason", step=1))

        max_firings = (self.max_iterations * 2) + 2
        self._run_until_session_finished(sid, max_firings=max_firings)

        # Check for connection failure during run
        if sid in self.session_errors:
            err = self.session_errors.pop(sid)
            print(f"\n⚠️ [Graceful Failure] Gemini API connection failure: {err}")
            print("  • Working Memory will NOT be persisted.")
            self.clear_session_facts(sid)
            return None

        # Retrieve final result
        final_fact = self._get_fact_by_kind("FinalAnswer", sid)
        final_answer = final_fact["text"] if final_fact else None

        self._print_fact_audit(sid)

        print("\n" + "=" * 80)
        print("🏁 Execution Complete")
        print("=" * 80)
        if final_answer:
            print(f"Final Answer: {final_answer}\n")
            # Persist working memory (Gemini-deduced)
            session_facts = [
                f for f in self.net.facts.values()
                if f.get("session_id") == sid
            ]
            self.wm_store.save(question, session_facts)
        else:
            print("Finished without explicit FinalAnswer fact.")

        # Clear working memory upon Finished resolution
        self.clear_session_facts(sid)

        return final_answer


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    try:
        gemini_llm = create_gemini_llm(model="gemini-2.5-flash", temperature=0)
    except Exception as e:
        print(f"⚠️ [Graceful Failure] Gemini API does not exist or cannot connect: {e}")
        gemini_llm = None

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
        "What is the total retail purchase price (MSRP) of a new entry-level Apple iPad Pro M4 in USD? "
        "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
    )


if __name__ == "__main__":
    main()
