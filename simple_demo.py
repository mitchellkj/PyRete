from py_rete.common import V
from py_rete.fact import Fact
from py_rete.network import ReteNetwork
from py_rete.production import Production


# =====================================================================
# 1. Traffic Lights Demo (Kept for Reference)
# =====================================================================

@Production(V("fact") << Fact(light_color="red"))
def make_green(net, fact):
    print("making green")
    fact["light_color"] = "green"
    net.update_fact(fact)


@Production(V("fact") << Fact(light_color="green"))
def make_red(net, fact):
    print("making red")
    fact["light_color"] = "red"
    net.update_fact(fact)


def lights_demo():
    print("=== Running Traffic Lights Demo ===")
    light_net = ReteNetwork()
    f1 = Fact(light_color="red")
    light_net.add_fact(f1)
    light_net.add_production(make_green)
    light_net.add_production(make_red)
    light_net.run(5)


# =====================================================================
# 2. ReACT Agent Tools Demo
# =====================================================================

class Tool(Fact):
    """Fact representing an available agent Tool in working memory."""

    def __init__(self, name=None, description=None, func=None, **kwargs):
        init_kwargs = {}
        if name is not None:
            init_kwargs["name"] = name
        if description is not None:
            init_kwargs["description"] = description
        if func is not None:
            init_kwargs["func"] = func
        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)


def stub_duckduck(query: str = "") -> str:
    print("Executing Tool duckduck")
    return f"Search results for: {query}"


def stub_calculator(expression: str = "") -> str:
    print("Executing Tool calculator")
    return f"Calculated result for: {expression}"


@Production(V("tool") << Tool())
def RegisterTool(net, tool):
    print(f"[RegisterTool] Acknowledged tool: '{tool['name']}' - {tool['description']}")
    net.remove_fact(tool)


def react_demo():
    print("\n=== Running ReACT Tool Registration Demo ===")
    net = ReteNetwork()
    net.add_production(RegisterTool)

    # 1. DuckDuckGo search tool
    search_tool = Tool(
        name="duckduck",
        description="A web search engine. Use this to as a search engine for general queries.",
        func=stub_duckduck,
    )

    # 2. Calculator tool (analogous to llm-math)
    calc_tool = Tool(
        name="calculator",
        description="A calculator tool for math expressions.",
        func=stub_calculator,
    )

    # Deposit the 2 tools into working memory
    net.add_fact(search_tool)
    net.add_fact(calc_tool)

    # Run the network to process rule activations
    net.run()


# =====================================================================
# Main Execution
# =====================================================================

def main():
    lights_demo()
    react_demo()


if __name__ == "__main__":
    main()
