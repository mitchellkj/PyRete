We want to create a demonstration of ReACT agent control being governed by
pyrete rules engine control rather than LangChain. We will build our code in
/Users/mitchellkj19/veytel/pulmoviz-python/PyReteDemo.  The objective is to
mirror a demo from Hand-On Large Language Models by Alammar et al.  In this publications
the authors introduce a LangChain coordinator that is triggered by a user request to
find the current cost of an iPad or other hardware component in a preferred currency
demomination.  "What is the latest iPad price in Euros?"  We assume tools for Search Engine
and Calculator.  Initial just stub those tools for iPad and Euros.  We want to first
examine how pyrete can drive LLM agents. We will use gemini as our LLM but we need to
figure out where the API key is in our environment. We have purchased a gemini API key and
have it stored in our secretes manager.  Try building the scaffolding for the demo before
we hook in actual tools.  Also let's assume we will run from a directory local venv under PyReteDemo.
Pick a name for that enviorment and provide a requirement.txt to pull in pyrete and anything else
we need to get running.


Something is wrong with getting PyCharm to recognize my python3.14 which appears to be
correct in the venv I created and am using in Terminal.
simple_demo.py	venv
mitchellkj19@Kevins-Mac-mini-2 PyRete % source venv/bin/activate
(venv) mitchellkj19@Kevins-Mac-mini-2 PyRete % ls
simple_demo.py	venv
(venv) mitchellkj19@Kevins-Mac-mini-2 PyRete % python --version
Python 3.14.7
(venv) mitchellkj19@Kevins-Mac-mini-2 PyRete %

However after creating the Project in PyCharm and pointing it at the $HOME/freelance/PyRete and using


It seems like PyCharm is latching on to an old virtual environment possibly from another project?
× This environment is externally managed
╰─> This Python installation is managed by uv and should not be modified.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
(python) mitchellkj19@Kevins-Mac-mini-2 PyRete % uv pip install py_rete google-genai python-dotenv langchain
  langchain-community duckduckgo-search numexpr
error: Failed to inspect Python interpreter from active virtual environment at `PyReteDemo/venv/lib/python/bin/python3`
  Caused by: Python interpreter not found at `/Users/mitchellkj19/freelance/PyRete/PyReteDemo/venv/lib/python/bin/python3`
zsh: command not found: langchain-community
(python) mitchellkj19@Kevins-Mac-mini-2 PyRete %

The actual venv is in /Users/mitchellkj19/freelance/PyRete/venv which I can activate and see the python3.12 is being
used via the Terminal tool.  I have no idea why PyCharm is fishing back to another project and venv?

Terrific.  The light demo works.  Now let's explore.  En route to our ultimate goal with this demo we want to
represent a set of Tools.  Write a class that extends Fact called Tool.  A Tool Fact will have a name, description, and
func attribute.  The func will actually point to a local python function that can run the tool but for now just
stub the func functions to say "Executing Tool {name}"
Create a new top level tester function called react_demo that will create it's own ReteNetwork deposit 2 Tools
for DuckDuckGo search and Calculator.  Use this code as a reference.
from langchain.agents import load_tools, Tool
from langchain.tools import DuckDuckGoSearchResults

# You can create the tool to pass to an agent
search = DuckDuckGoSearchResults()
search_tool = Tool(
    name="duckduck",
    description="A web search engine. Use this to as a search engine for general queries.",
    func=search.run,
)

# Prepare tools
tools = load_tools(["llm-math"], llm=openai_llm)
tools.append(search_tool)
The first iteration will merely define a Tool Fact make two of them and proivde a single Production that
awakes againt a Tool and acknowleges it.  Let's call the Production RegisterTool.

Go ahead and make the code changes as we have committed the previous branch.  Also keep the lights demo in the
code for reference.

Now let's try to fork up functions for Tools.  To achieve this we will write a new Production RunTool that will
use lambda like calling from the func attribute.  But while we match and interrogate tools we don't want
@Production(V("tool") << Tool())
def RegisterTool(net, tool):
    print(f"[RegisterTool] Acknowledged tool: '{tool['name']}' - {tool['description']}")
    net.remove_fact(tool)
To run again.  So let's try to create a state attribute on Tool the can be "Registering" | "Running"
Change RegisterTool to Filter on only Tools with the Registering state, and rather than retract use update to
the new state of Running.   RunTool Production will call the helper and retract the Tool.  This sequence will
let us explore py_rete Filter, python lambda like reflectivity.  Among other things.

Before we continue exploring possibilities with py_rete coordinating a LLM which in turns coordinates a set of
tools.  Let's try to work with LangChain to do the same thing.  We have created a react_langchain.py that
uses LangChain coordinate.  Currently react_langchain.py uses OpenAI and expects a API_KEY.  We want to swap in
Google Gemi and pull the AI Key from our SecretsManager.  For example.

def get_gemini_response(question: str, prompt: List[str]) -> str:
    """
    Load Google model and provide Mongo query as response.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([prompt[0], question])
    query_str = response.text.strip()  # Strip whitespace for cleaner parsing
    logger.info(f"[agentic:get_gemini] Generated query for '{question[:50]}...': {query_str}")
    return query_str

Lock at ./backend/agentic/agentic.py and agentic_endpoint.py and set up react_langchain.py to use Gemini with our
API_KEY.  Then clean up the code in react_langchain.py creating a __main__ entry point.  Create a requirements.txt
for all imports.  Include those needed by simple_demo.py

Can you help me understand LangChain here?
Gemini chooses some sites to make the query in a RAG like fashion but why is Wikipedia failing on the
 DDGSException('ConnectError: ConnectError(\'error sending request...  What is DDGS?  I then asks
 grokipedia and a duck duck site.  So it's looking in three places but not Apple?
 It seems like so much info is not visible like how many cycles did it run?
 /Users/mitchellkj19/freelance/PyRete/venv/bin/python /Users/mitchellkj19/freelance/PyRete/react_langchain.py
======================================================================
🤖 LangChain ReAct Agent (Gemini + DuckDuckGo + Math Tools)
======================================================================
2026-08-20 17:00:19,548 [INFO] Fetching secret 'gemini-api-key' from GCP project 'veytel-cloud-store'...
2026-08-20 17:00:21,552 [INFO] API key successfully retrieved from Secret Manager (39 chars).
2026-08-20 17:00:21,628 [INFO] NumExpr defaulting to 8 threads.

User Query: What is the current price of a MacBook Pro in USD? How much would it cost in EUR if the exchange rate is 0.85 EUR for 1 USD?



> Entering new AgentExecutor chain...
Action: duckduck
Action Input: current price of MacBook Pro in USD2026-08-20 17:00:23,242 [INFO] Error in engine wikipedia: DDGSException('ConnectError: ConnectError(\'error sending request for url (https://wt.wikipedia.org/w/api.php?action=opensearch&profile=fuzzy&limit=1&search=current%20price%20of%20MacBook%20Pro%20in%20USD) > client error (Connect) > dns error > DNS error: no records found for Query { name: Name("wt.wikipedia.org."), query_type: A, query_class: IN }\')')
2026-08-20 17:00:23,571 [INFO] response: https://grokipedia.com/api/typeahead?query=current+price+of+MacBook+Pro+in+USD&limit=1 200
2026-08-20 17:00:24,713 [INFO] HTTP Request: POST https://html.duckduckgo.com/html/ "HTTP/2 200 OK"
snippet: The original MSRP price in the United States for all MacBook Pro models; 2006-Current. EveryMac.com USA Mac Pricing always has been diligently hand compiled and configured., title: Original Prices for All MacBook Pro in the United States (2006-Current ..., link: https://everymac.com/global-mac-prices/all-msrp-macbook-pro-prices-in-united-states.html, snippet: Steve H Posted on August 13, 2026 Our Apple award-winning MacBook Price Trackers are continually updated with the latest information on prices, bundles, and availability for the new MacBook Neo, 16″ and 14″ M5, M5 Pro, & M5 Max MacBook Pros, along with 13″ and 15″ M5 MacBook Airs from Apple's authorized online retailers., title: Updated MacBook Price Trackers, link: https://www.macprices.net/2026/08/13/updated-macbook-price-trackers-9/, snippet: As of mid-2024, the Apple MacBook Pro starts at $1,699 for the 14-inch M4 base model, but real-world cost ranges from $1,699 to over $7,800 depending on configuration, region, and purchase channel—making 'MacBook Pro cost' one of the most context-dependent tech pricing questions today., title: How Much Does a MacBook Pro Cost in 2024-2025?, link: https://electronics.alibaba.com/question/macbook-pro-cost-breakdown-real-prices-what-you-actually-pay, snippet: As of November 9, 2025, Apple's MacBook lineup includes the lightweight MacBook Air with the M4 chip and the more powerful MacBook Pro series featuring M5, M4 Pro, and M4 Max chips. The latest updates include the March 2025 refresh of the MacBook Air with M4 (introducing a new Sky Blue color and a starting price of $999) and the October 2025 release of the base 14-inch MacBook Pro with M5 ..., title: The Apple MacBook Price List in the USA 2026 | Rank1one, link: https://rank1one.com/apple-macbook-price-list/Action: Calculator
Action Input: 1699 * 0.85Answer: 1444.1499999999999I now know the final answer
Final Answer: The current starting price for a MacBook Pro (14-inch M4 base model) is $1,699 USD. This would cost approximately 1444.15 EUR.

> Finished chain.

======================================================================
🎯 Final Agent Output:
The current starting price for a MacBook Pro (14-inch M4 base model) is $1,699 USD. This would cost approximately 1444.15 EUR.
======================================================================

Process finished with exit code 0
