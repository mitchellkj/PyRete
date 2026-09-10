#!/usr/bin/env python3
"""
react_langchain_verbose.py: ReAct agent powered by LangChain and Google Gemini
with maximum verbose logging, debug tracing, and step inspection enabled.
"""

import os
import logging
import warnings
from typing import Optional

# Enable debug logging across standard library and LangChain components
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy deprecation warnings while retaining debug traces
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")

# Force LangChain global environment flags for maximum verbosity
os.environ["LANGCHAIN_VERBOSE"] = "true"
os.environ["LANGCHAIN_DEBUG"] = "true"

try:
    from langchain_core.globals import set_debug, set_verbose
    set_debug(True)
    set_verbose(True)
except ImportError:
    try:
        import langchain
        langchain.debug = True
        langchain.verbose = True
    except (ImportError, AttributeError):
        pass

try:
    from langchain_community.agent_toolkits.load_tools import load_tools
except ImportError:
    from langchain.agents import load_tools

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

try:
    from langchain_community.tools import DuckDuckGoSearchResults
except ImportError:
    from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_api_key(
        secret_id: str = "gemini-api-key",
        version: str = "latest",
        project_id: str = "veytel-cloud-store",
) -> str:
    """
    Retrieve Gemini API key from environment variables (GEMINI_API_KEY / GOOGLE_API_KEY)
    or directly from Google Cloud Secret Manager if not present in the environment.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        logger.info("Found Gemini API key in environment variables.")
        return api_key.strip()

    logger.info(f"Fetching secret '{secret_id}' from GCP project '{project_id}'...")
    try:
        from google.cloud import secretmanager

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8").strip()
        logger.info(f"API key successfully retrieved from Secret Manager ({len(api_key)} chars).")

        # Cache in environment for subprocesses and downstream libraries
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        return api_key
    except Exception as e:
        logger.error(f"Failed to fetch secret from Secret Manager: {e}")
        raise ValueError(
            f"GEMINI_API_KEY not found in environment and Secret Manager lookup failed: {e}"
        )


def build_react_agent(
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
) -> AgentExecutor:
    """
    Constructs and returns the ReAct agent executor configured with Gemini and tools,
    with all debug and verbose execution parameters activated.
    """
    if not api_key:
        api_key = get_gemini_api_key()

    # 1. Initialize Gemini LLM with verbose callbacks
    gemini_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
        verbose=True,
    )

    # 2. Define ReAct Prompt Template
    react_template = """Answer the following questions as best you can. You have access to the following tools:

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

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=react_template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    )

    # 3. Create Tools (DuckDuckGo Search + Math)
    search = DuckDuckGoSearchResults()
    search_tool = Tool(
        name="duckduck",
        description="A web search engine. Use this to search the web for general queries, latest prices, and news.",
        func=search.run,
        verbose=True,
    )

    tools = load_tools(["llm-math"], llm=gemini_llm)
    for tool in tools:
        tool.verbose = True
    tools.append(search_tool)

    # 4. Construct ReAct agent & executor with verbose & intermediate step tracking
    agent = create_react_agent(gemini_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor


def main():
    """Main execution entry point."""
    print("=" * 70)
    print("🤖 LangChain ReAct Agent [VERBOSE DEBUG MODE] (Gemini + Tools)")
    print("=" * 70)

    try:
        agent_executor = build_react_agent()
        query = (
            "What is the current price of a MacBook Pro in USD? "
            "How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?"
        )
        print(f"\nUser Query: {query}\n")
        response = agent_executor.invoke({"input": query})

        # Detailed inspection of intermediate steps
        intermediate_steps = response.get("intermediate_steps", [])
        if intermediate_steps:
            print("\n" + "-" * 70)
            print(f"📋 Intermediate ReAct Steps ({len(intermediate_steps)} total):")
            print("-" * 70)
            for idx, (action, observation) in enumerate(intermediate_steps, start=1):
                print(f"Step {idx}:")
                print(f"  • Tool       : {action.tool}")
                print(f"  • Tool Input : {action.tool_input}")
                print(f"  • Log / Thought: {action.log.strip()}")
                print(f"  • Observation: {observation}\n")

        print("\n" + "=" * 70)
        print("🎯 Final Agent Output:")
        print(response.get("output", response))
        print("=" * 70)
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
