#!/usr/bin/env python3
"""
backend/coordinator.py: PyRete ReAct coordinator engine for the FastAPI backend.
Features:
  - Session-scoped Working Memory: Facts maintain session cookies/identifiers (`session_id`)
    allowing concurrent multi-user queries to live side-by-side in working memory without interference.
  - Robust Fact Hashability: Enforces that all Fact attributes are hashable, preventing
    py_rete WME hash failures (TypeError: unhashable type: 'list') when Gemini outputs structured content parts.
  - Goal session-tagging and session-scoped production rules.
  - Persistent Working Memory Store keyed by prompt / User Query hash (SHA-256).
  - First checks for cached Working Memory; if present, unmarshals facts and substitutes the
    current session id with the persisted one, returning immediately on Finished fact detection.
  - Working Memory is cleanly cleared upon Finished resolution.
  - Checks Gemini API accessibility before inference: if accessible, lets ReAct execute through
    Gemini and persists the deduced facts; otherwise fails gracefully and does NOT persist WM.
"""

import os
import re
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from py_rete.common import V
from py_rete.fact import Fact
from py_rete.network import ReteNetwork
from py_rete.production import Production

from tool import Tool, DuckDuckGoSearcher, Calculator

logger = logging.getLogger("coordinator")


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
                # Standard LangChain / Gemini block format: {'type': 'text', 'text': '...'}
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
# 2. Prompt Template & Parser
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

    if "Final Answer:" in cleaned:
        parts = cleaned.split("Final Answer:", 1)
        thought_part = parts[0].replace("Thought:", "").strip()
        final_answer = parts[1].strip()
        return {
            "type": "final_answer",
            "thought": thought_part,
            "final_answer": final_answer,
        }

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


def create_llm_backend(model: Optional[str] = None, temperature: float = 0.0) -> Any:
    """Creates the Gemini LLM backend. Assumes Gemini API accessibility."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise ConnectionError(
            "Gemini API key not found in environment (GEMINI_API_KEY or GOOGLE_API_KEY) "
            "and Secret Manager lookup failed."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    logger.info(f"Initialized Gemini LLM backend using model '{model_name}'.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


# =============================================================================
# 4. Working Memory Persistence
# =============================================================================

def get_default_wm_dir() -> str:
    """Returns the directory used for persistent working memory storage."""
    if "WM_STORAGE_DIR" in os.environ:
        return os.environ["WM_STORAGE_DIR"]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    if os.path.basename(base_dir) == "backend":
        return os.path.join(parent_dir, "wm_storage")
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
        """Computes a SHA-256 hash of the normalized prompt or User Query."""
        normalized = query.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_file_path(self, query_hash: str) -> str:
        return os.path.join(self.storage_dir, f"{query_hash}.json")

    def exists(self, query: str) -> bool:
        """Checks if persisted working memory exists for this prompt or user query."""
        query_hash = self.compute_query_hash(query)
        path = self.get_file_path(query_hash)
        return os.path.exists(path)

    def load(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Loads persisted working memory facts if they exist."""
        query_hash = self.compute_query_hash(query)
        path = self.get_file_path(query_hash)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("facts", [])
        except Exception as e:
            logger.error(f"Failed to load persisted working memory from '{path}': {e}")
            return None

    def save(self, query: str, facts: List[Fact]) -> bool:
        """
        Persists episodic working memory facts to disk keyed by prompt / user query hash.
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
# 5. Session-Aware PyRete Coordinator Engine
# =============================================================================

class PyReteCoordinatorEngine:
    """
    Goal-driven forward chaining rule coordinator using PyRete.
    Maintains session integrity across concurrent requests by tagging Goals
    and episodic facts with `session_id` as a frequency dimension.
    Guarantees all facts in working memory contain hashable attributes.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 8,
        storage_dir: Optional[str] = None,
    ):
        self.max_iterations = max_iterations
        self.wm_store = WorkingMemoryStore(storage_dir=storage_dir)

        default_tools = [
            DuckDuckGoSearcher(
                name="duckduck",
                description="A web search engine. Use this to find retail purchase prices, starting MSRP, and hardware specs.",
                max_results=5,
            ),
            Calculator(name="Calculator"),
        ]
        self.tools = {tool.name: tool for tool in (tools or default_tools)}

        # Long-lived PyRete network with compiled production rules
        self.net = ReteNetwork()
        for tool in self.tools.values():
            self.net.add_fact(tool)

        # Track session-scoped execution errors
        self.session_errors: Dict[str, str] = {}

        # LLM backend management
        self._llm = llm
        self.llm_init_error: Optional[str] = None
        if self._llm is None:
            try:
                self._llm = create_llm_backend()
            except Exception as e:
                logger.warning(f"Initial Gemini LLM backend creation deferred: {e}")
                self.llm_init_error = str(e)

        # Compile and register productions ONCE into self.net
        self._register_productions()

    def _get_or_create_llm(self) -> Optional[Any]:
        """Lazy loader / accessor for the Gemini LLM backend."""
        if self._llm is not None:
            return self._llm
        try:
            self._llm = create_llm_backend()
            self.llm_init_error = None
            return self._llm
        except Exception as e:
            self.llm_init_error = str(e)
            return None

    def check_gemini_accessibility(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies that Gemini API exists and can connect.
        Returns (True, None) if accessible, or (False, error_message).
        Explicitly distinguishes quota exhaustion (429) from connectivity errors.
        """
        llm = self._get_or_create_llm()
        if llm is None:
            err = self.llm_init_error or "Gemini API key not found in environment or Secret Manager."
            return False, err

        try:
            response = llm.invoke("ping")
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

    def _extract_session_audit(self, session_id: str) -> List[Dict[str, Any]]:
        """Extracts ordered audit items for a specific session_id."""
        audit = []
        for f in self.net.facts.values():
            if f.get("session_id") == session_id:
                kind = f.get("kind")
                if kind in ("Thought", "Action", "ActionInput", "Observation", "FinalAnswer"):
                    content = f.get("text") or f.get("name") or ""
                    audit.append({
                        "step": f.get("step"),
                        "kind": kind,
                        "content": content,
                    })
        audit.sort(key=lambda x: (x["step"] if x["step"] is not None else 999))
        return audit

    def _build_scratchpad(self, session_id: str, up_to_step: int) -> str:
        """Assembles previous steps for a specific session_id into ReAct scratchpad."""
        parts = []
        for s in range(1, up_to_step):
            thought = next((f for f in self.net.facts.values() if f.get("kind") == "Thought" and f.get("session_id") == session_id and f.get("step") == s), None)
            action = next((f for f in self.net.facts.values() if f.get("kind") == "Action" and f.get("session_id") == session_id and f.get("step") == s), None)
            action_input = next((f for f in self.net.facts.values() if f.get("kind") == "ActionInput" and f.get("session_id") == session_id and f.get("step") == s), None)
            observation = next((f for f in self.net.facts.values() if f.get("kind") == "Observation" and f.get("session_id") == session_id and f.get("step") == s), None)

            th_text = f" {thought['text']}" if thought and thought.get("text") else ""
            act_text = action["name"] if action else ""
            inp_text = action_input["text"] if action_input else ""
            obs_text = observation["text"] if observation else ""

            parts.append(f"{th_text}\nAction: {act_text}\nAction Input: {inp_text}\nObservation: {obs_text}\nThought:")
        return "".join(parts)

    def _register_productions(self):
        """
        Compiles and registers session-scoped Reason and Act productions once.
        Using `session_id=V('sid')` forces all join nodes to match only within the same session frequency.
        """

        # ---------------------------------------------------------------------
        # Rule 1: Reason Production (session-scoped)
        # ---------------------------------------------------------------------
        @Production(
            (V("g") << Goal(goal_type="Reason", session_id=V("sid")))
            & (V("tracker") << StepTracker(session_id=V("sid")))
            & (V("q") << Question(session_id=V("sid")))
        )
        def reason_production(net, g, tracker, q, sid):
            current_step = tracker.get("current_step", 1)
            max_steps = tracker.get("max_steps", self.max_iterations)
            logger.info(f"[Session {sid}] >>> Reason Production triggered (step {current_step}/{max_steps}).")

            if current_step > max_steps:
                logger.warning(
                    f"[Session {sid}] Reached maximum iterations ({current_step} > {max_steps}). Terminating with Finished fact."
                )
                net.remove_fact(g)
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            tools_desc = "\n".join(f"{t.name}: {t.description}" for t in self.tools.values())
            tool_names = ", ".join(self.tools.keys())
            scratchpad = self._build_scratchpad(session_id=sid, up_to_step=current_step)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                tool_names=tool_names,
                question=q.get("text", ""),
                scratchpad=scratchpad,
            )

            logger.info(
                f"[Session {sid}] Sending prompt to Gemini ({len(prompt)} chars, scratchpad_len={len(scratchpad)} chars)..."
            )

            try:
                llm = self._get_or_create_llm()
                response = llm.invoke(prompt)
                # Safely extract text string to prevent unhashable list/dict WMEs
                pred_text = extract_text_from_response(response)
                logger.info(
                    f"[Session {sid}] Gemini response received ({len(pred_text)} chars): {repr(pred_text[:120].strip())}..."
                )
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
                    logger.error(f"[Session {sid}] Gemini API invocation error: {e}", exc_info=True)
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

            net.remove_fact(g)
            net.add_fact(LLMPrediction(session_id=sid, text=pred_text, step=current_step))
            net.add_fact(Goal(session_id=sid, goal_type="Act", step=current_step))
            logger.info(f"[Session {sid}] Reason phase complete. Transitioned to Goal(goal_type='Act', step={current_step}).")

        # ---------------------------------------------------------------------
        # Rule 2: Act Production (session-scoped)
        # ---------------------------------------------------------------------
        @Production(
            (V("g") << Goal(goal_type="Act", session_id=V("sid")))
            & (V("pred") << LLMPrediction(session_id=V("sid")))
            & (V("tracker") << StepTracker(session_id=V("sid")))
        )
        def act_production(net, g, pred, tracker, sid):
            current_step = tracker.get("current_step", 1)
            raw_text = pred.get("text", "")
            logger.info(f"[Session {sid}] >>> Act Production triggered (step {current_step}).")

            net.remove_fact(g)
            net.remove_fact(pred)

            parsed = parse_react_response(raw_text)
            logger.info(f"[Session {sid}] Parsed response type='{parsed['type']}'")

            if parsed["type"] == "final_answer":
                thought_text = parsed.get("thought", "")
                final_answer_text = parsed.get("final_answer", "")
                logger.info(f"[Session {sid}] Final Answer reached: {repr(final_answer_text[:120])}")
                if thought_text:
                    net.add_fact(Thought(session_id=sid, text=thought_text, step=current_step))
                net.add_fact(FinalAnswer(session_id=sid, text=final_answer_text, step=current_step))
                net.add_fact(Finished(session_id=sid, step=current_step))
                net.add_fact(Goal(session_id=sid, goal_type="Finished", step=current_step))
                return

            thought_text = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", "")
            logger.info(
                f"[Session {sid}] Step {current_step} Action='{action_name}', "
                f"Action Input='{action_input[:100]}', Thought='{thought_text[:80]}'"
            )

            if thought_text:
                net.add_fact(Thought(session_id=sid, text=thought_text, step=current_step))
            net.add_fact(Action(session_id=sid, name=action_name, step=current_step))
            net.add_fact(ActionInput(session_id=sid, text=action_input, step=current_step))

            # Robust tool lookup with case-insensitivity fallback
            selected_tool = self.tools.get(action_name)
            if not selected_tool:
                for t_name, t_inst in self.tools.items():
                    if t_name.lower() == action_name.lower():
                        selected_tool = t_inst
                        break

            if selected_tool:
                logger.info(f"[Session {sid}] Executing tool '{selected_tool.name}' with input: {action_input[:100]}")
                obs_str = selected_tool.run(action_input)
                logger.info(
                    f"[Session {sid}] Tool '{selected_tool.name}' returned observation ({len(obs_str)} chars): "
                    f"{repr(obs_str[:120].strip())}..."
                )
            else:
                available = ", ".join(self.tools.keys())
                obs_str = f"Error: Tool '{action_name}' not found. Available: [{available}]."
                logger.warning(f"[Session {sid}] Tool '{action_name}' not found. Available: [{available}]")

            net.add_fact(Observation(session_id=sid, text=obs_str, step=current_step))

            next_step = current_step + 1
            tracker["current_step"] = next_step
            net.update_fact(tracker)
            net.add_fact(Goal(session_id=sid, goal_type="Reason", step=next_step))
            logger.info(f"[Session {sid}] Act phase complete. Advanced to step {next_step}. Asserted Goal(Reason).")

        self.net.add_production(reason_production)
        self.net.add_production(act_production)

    def _run_until_session_finished(self, session_id: str, max_firings: int) -> int:
        """Executes rule firings prioritizing the specified session until it finishes."""
        firings = 0
        logger.info(f"[Session {session_id}] Starting Rete execution loop (max_firings={max_firings})...")
        while firings < max_firings:
            if self._is_session_finished(session_id):
                logger.info(f"[Session {session_id}] Finished fact detected after {firings} firing(s).")
                break

            matches = list(self.net.matches)
            if not matches:
                session_facts = [
                    f"{f.get('kind')}(step={f.get('step')})"
                    for f in self.net.facts.values()
                    if f.get("session_id") == session_id
                ]
                logger.warning(
                    f"[Session {session_id}] No active rule matches in Rete network after {firings} firing(s)! "
                    f"Session facts currently in working memory: {session_facts}"
                )
                break

            # Prioritize matches for this session
            session_matches = [
                m for m in matches
                if m.token.binding.get(V("sid")) == session_id
            ]
            if not session_matches:
                logger.warning(
                    f"[Session {session_id}] {len(matches)} total matches in net, but 0 matches for session {session_id}."
                )

            chosen = session_matches[0] if session_matches else matches[0]
            prod_name = getattr(chosen.pnode.production, "__name__", str(chosen.pnode.production))
            logger.info(
                f"[Session {session_id}] Firing rule #{firings + 1}: '{prod_name}' "
                f"(total matches={len(matches)}, session matches={len(session_matches)})"
            )
            chosen.fire()
            firings += 1

        if firings >= max_firings:
            logger.warning(f"[Session {session_id}] Reached maximum rule firings limit ({max_firings}).")

        return firings

    def run_detailed(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes query following the session-scoped and persisted working memory strategy:
          1. Generates or preserves session_id for session integrity.
          2. Clears any pre-existing facts for session_id in working memory.
          3. Checks for cached Working Memory (keyed by query hash):
             - If found, unmarshals facts, substitutes the current session_id,
               inserts into network, and immediately returns since Finished fact is present.
             - Clears working memory upon Finished resolution.
          4. If cached WM does not exist:
             - Checks Gemini API accessibility.
             - If NOT accessible: fails gracefully and does NOT persist Working Memory.
             - If accessible: runs ReAct loop through Gemini.
             - If successful, persists deduced Working Memory keyed by query hash.
             - Clears working memory upon Finished resolution.
        """
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        query_hash = self.wm_store.compute_query_hash(question)

        # Clear any prior state or leftover facts for this session to ensure a clean slate
        self.clear_session_facts(sid)
        self.session_errors.pop(sid, None)

        # ---------------------------------------------------------------------
        # Step 1: Check for Persisted Working Memory
        # ---------------------------------------------------------------------
        if self.wm_store.exists(question):
            logger.info(f"[Session {sid}] Persisted WM hit for query hash {query_hash}. Unmarshaling...")
            persisted_facts = self.wm_store.load(question)
            if persisted_facts:
                # Unmarshal and substitute current session_id
                for item in persisted_facts:
                    item_copy = dict(item)
                    item_copy["session_id"] = sid
                    kind = item_copy.get("kind")
                    cls = FACT_CLASS_MAP.get(kind, BaseReActFact)
                    kwargs = {k: v for k, v in item_copy.items() if k != "kind"}
                    self.net.add_fact(cls(**kwargs))

                # If Finished fact is present in the restored WM, return immediately
                if self._is_session_finished(sid):
                    final_fact = self._get_fact_by_kind("FinalAnswer", sid)
                    final_answer = final_fact["text"] if final_fact else None
                    audit = self._extract_session_audit(sid)

                    # Clear working memory upon Finished resolution
                    self.clear_session_facts(sid)

                    logger.info(
                        f"[Session {sid}] Finished fact detected in restored WM. Immediately returning."
                    )
                    return {
                        "success": True if final_answer else False,
                        "session_id": sid,
                        "final_answer": final_answer or "Completed from persisted working memory.",
                        "audit": audit,
                        "status": "completed",
                        "cached": True,
                        "query_hash": query_hash,
                    }

        # ---------------------------------------------------------------------
        # Step 2: Persisted WM does not exist -> Check Gemini Accessibility
        # ---------------------------------------------------------------------
        is_accessible, access_error = self.check_gemini_accessibility()
        if not is_accessible:
            err_msg = access_error or "Gemini API is unavailable or cannot connect."
            logger.error(f"[Session {sid}] Gemini accessibility check failed: {err_msg}")
            self.clear_session_facts(sid)
            return {
                "success": False,
                "session_id": sid,
                "final_answer": f"Unable to process query: {err_msg}",
                "audit": [],
                "status": "api_unavailable",
                "error": err_msg,
                "cached": False,
                "query_hash": query_hash,
            }

        # ---------------------------------------------------------------------
        # Step 3: Let the ReAct process execute through Gemini
        # ---------------------------------------------------------------------
        logger.info(f"[Session {sid}] Starting live Gemini ReAct loop for: {question[:60]}...")
        self.net.add_fact(Question(session_id=sid, text=question, step=0))
        self.net.add_fact(StepTracker(session_id=sid, current_step=1, max_steps=self.max_iterations, step=0))
        self.net.add_fact(Goal(session_id=sid, goal_type="Reason", step=1))

        initial_matches = list(self.net.matches)
        logger.info(
            f"[Session {sid}] Seeded initial facts in Rete net (total net facts={len(self.net.facts)}). "
            f"Active rule activations ready to fire: {len(initial_matches)}"
        )

        max_firings = (self.max_iterations * 2) + 2
        firings = self._run_until_session_finished(sid, max_firings=max_firings)
        logger.info(f"[Session {sid}] ReAct execution loop completed with {firings} firing(s).")

        # Check if execution failed during the loop
        if sid in self.session_errors:
            err = self.session_errors.pop(sid)
            logger.error(f"[Session {sid}] Execution halted due to error: {err}")
            # Collect any intermediate facts accumulated before the failure
            partial_audit = self._extract_session_audit(sid)
            self.clear_session_facts(sid)
            return {
                "success": False,
                "session_id": sid,
                "final_answer": f"Unable to process query: {err}",
                "audit": partial_audit,
                "status": "connection_failed",
                "error": err,
                "cached": False,
                "query_hash": query_hash,
            }

        final_fact = self._get_fact_by_kind("FinalAnswer", sid)
        final_answer = final_fact["text"] if final_fact else None
        audit = self._extract_session_audit(sid)

        if final_answer:
            # Collect current session facts to persist (Gemini-deduced)
            session_facts = [
                f for f in self.net.facts.values()
                if f.get("session_id") == sid
            ]
            self.wm_store.save(question, session_facts)
            status = "completed"
            success = True
            logger.info(f"[Session {sid}] Final answer produced ({len(final_answer)} chars). Working memory cached.")
        else:
            success = False
            observations = [
                item for item in audit
                if item["kind"] == "Observation" and item.get("content")
            ]
            thoughts = [
                item for item in audit
                if item["kind"] == "Thought" and item.get("content")
            ]
            if firings == 0:
                status = "no_rules_matched"
                final_answer = (
                    "Execution halted: no rule productions matched the initial working memory state. "
                    "Check server logs for diagnostic details."
                )
                logger.error(f"[Session {sid}] Execution halted: 0 rule firings occurred.")
            else:
                status = "exhausted"
                logger.warning(
                    f"[Session {sid}] Search limit reached after {self.max_iterations} steps "
                    f"({firings} firings). No FinalAnswer fact was produced."
                )
                partial_lines = [
                    f"Search iteration limit reached after {self.max_iterations} steps. "
                    "No definitive Final Answer was produced. "
                    "Below is a summary of intermediate findings:"
                ]
                if thoughts:
                    last_thought = thoughts[-1]["content"]
                    partial_lines.append(f"\nLast reasoning: {last_thought[:300]}")
                if observations:
                    partial_lines.append(f"\nIntermediate observations ({len(observations)} collected):")
                    for obs in observations[-3:]:  # Show the last 3 observations
                        snippet = obs["content"][:200].replace("\n", " ")
                        partial_lines.append(f"  • Step {obs.get('step', '?')}: {snippet}")
                if not thoughts and not observations:
                    partial_lines.append("\nNo intermediate facts were recorded.")
                final_answer = "\n".join(partial_lines)

        # Clear working memory upon Finished resolution
        self.clear_session_facts(sid)

        return {
            "success": success,
            "session_id": sid,
            "final_answer": final_answer,
            "audit": audit,
            "status": status,
            "cached": False,
            "query_hash": query_hash,
        }
