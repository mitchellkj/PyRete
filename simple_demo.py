from py_rete.common import V
from py_rete.fact import Fact
from py_rete.network import ReteNetwork
from py_rete.production import Production

# 1. Define initial fact
f1 = Fact(light_color="red")


# 2. Define production rules
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


def light_demo():
    # 3. Instantiate the network
    light_net = ReteNetwork()

    # 4. Add fact and productions
    light_net.add_fact(f1)
    light_net.add_production(make_green)
    light_net.add_production(make_red)

    # 5. Run the network (fires up to 5 rule activations)
    light_net.run(5)


def main():
    light_demo()


if __name__ == "__main__":
    main()
