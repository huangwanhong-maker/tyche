"""Cross-check values for tyche::networks from networkx.

Run with networkx installed (any scratch environment):

    python make_networkx_fixtures.py > networkx_networks.txt

Three networks are written with networkx's statistics of them: Zachary's karate club (undirected; its two
clubs as a category), a seeded random directed network with a category and a number on each node, and a
seeded random undirected network with weights. Lines:

    network <name> <directed 0|1> <weighted 0|1> <nodes> <a-b:w;...>
    category <name> <c_0,c_1,...>            value <name> <x_0,x_1,...>
    scalar <name> <statistic> <value>         vector <name> <statistic> <v_0,v_1,...>
    census <name> <the sixteen counts in the order 003, 012, 102, 021D, ..., 300>

These are cross-checks, not references: the references of the checks are the closed forms, hand-worked
examples and brute force in check_network_descriptive.cpp and check_mixing.cpp.

Effective sizes are asked node by node (nodes=list(G)). networkx 3.7's fast path for all nodes at once,
taken when scipy is installed and `nodes` is None, disagrees with its own node-by-node definition (Burt's
formula, as its docstring states it) for directed and weighted graphs, e.g. 8.3333 against 8.2083 for node
2 of the random directed network here; the node-by-node values are the ones the docstring defines.
"""

import datetime
import random

import networkx as nx

NAMES = ["003", "012", "102", "021D", "021U", "021C", "111D", "111U", "030T", "030C", "201", "120D", "120U", "120C", "210", "300"]


def fmt(values):
    return ",".join(repr(float(v)) for v in values)


def write_network(name, G, weighted):
    edges = ";".join(f"{a}-{b}:{G[a][b].get('weight', 1.0)!r}" for a, b in G.edges())
    print("network", name, int(G.is_directed()), int(weighted), G.number_of_nodes(), edges)


def per_node(G, d):
    return [d[v] for v in range(G.number_of_nodes())]


print(f"# networkx {nx.__version__}; generated {datetime.date.today().isoformat()} by make_networkx_fixtures.py")

# ---- Zachary's karate club (Zachary 1977, J. Anthropol. Res. 33: 452-473), as networkx holds it ----
K = nx.karate_club_graph()
for a, b in K.edges():
    K[a][b].pop("weight", None)
write_network("karate", K, False)
club = [0 if K.nodes[v]["club"] == "Mr. Hi" else 1 for v in range(K.number_of_nodes())]
print("category", "karate", ",".join(str(c) for c in club))
nx.set_node_attributes(K, {v: club[v] for v in K}, "club01")
print("scalar", "karate", "transitivity", repr(nx.transitivity(K)))
print("scalar", "karate", "average_clustering", repr(nx.average_clustering(K)))
print("scalar", "karate", "degree_assortativity", repr(nx.degree_assortativity_coefficient(K)))
print("scalar", "karate", "attribute_assortativity", repr(nx.attribute_assortativity_coefficient(K, "club01")))
print("scalar", "karate", "density", repr(nx.density(K)))
print("vector", "karate", "clustering", fmt(per_node(K, nx.clustering(K))))
print("vector", "karate", "constraint", fmt(per_node(K, nx.constraint(K))))
print("vector", "karate", "effective_size", fmt(per_node(K, nx.effective_size(K, nodes=list(K)))))
print("vector", "karate", "core_number", fmt(per_node(K, nx.core_number(K))))
print("vector", "karate", "triangles", fmt(per_node(K, nx.triangles(K))))

# ---- a random directed network ----
D = nx.gnm_random_graph(30, 140, seed=20261113, directed=True)
rng = random.Random(20261113)
cat = [rng.randrange(3) for _ in range(30)]
val = [round(rng.gauss(0, 1), 6) for _ in range(30)]
nx.set_node_attributes(D, {v: cat[v] for v in D}, "cat")
nx.set_node_attributes(D, {v: val[v] for v in D}, "val")
write_network("digraph", D, False)
print("category", "digraph", ",".join(str(c) for c in cat))
print("value", "digraph", fmt(val))
census = nx.triadic_census(D)
print("census", "digraph", ",".join(str(census[n]) for n in NAMES))
print("scalar", "digraph", "reciprocity", repr(nx.overall_reciprocity(D)))
print("scalar", "digraph", "density", repr(nx.density(D)))
print("scalar", "digraph", "degree_assortativity_out_in", repr(nx.degree_assortativity_coefficient(D, x="out", y="in")))
print("scalar", "digraph", "degree_assortativity_in_in", repr(nx.degree_assortativity_coefficient(D, x="in", y="in")))
print("scalar", "digraph", "degree_assortativity_out_out", repr(nx.degree_assortativity_coefficient(D, x="out", y="out")))
print("scalar", "digraph", "attribute_assortativity", repr(nx.attribute_assortativity_coefficient(D, "cat")))
print("scalar", "digraph", "numeric_assortativity", repr(nx.numeric_assortativity_coefficient(D, "val")))
print("scalar", "digraph", "strong_components", repr(float(nx.number_strongly_connected_components(D))))
print("scalar", "digraph", "weak_components", repr(float(nx.number_weakly_connected_components(D))))
print("vector", "digraph", "constraint", fmt(per_node(D, nx.constraint(D))))
print("vector", "digraph", "effective_size", fmt(per_node(D, nx.effective_size(D, nodes=list(D)))))
U = D.to_undirected()
print("scalar", "digraph", "transitivity_underlying", repr(nx.transitivity(U)))
print("vector", "digraph", "core_number_underlying", fmt(per_node(U, nx.core_number(U))))

# ---- a sparse directed network with several strong components ----
S = nx.gnm_random_graph(40, 60, seed=7, directed=True)
write_network("sparse", S, False)
print("scalar", "sparse", "strong_components", repr(float(nx.number_strongly_connected_components(S))))
print("scalar", "sparse", "weak_components", repr(float(nx.number_weakly_connected_components(S))))
print("vector", "sparse", "strong_component_sizes", fmt(sorted(len(c) for c in nx.strongly_connected_components(S))))

# ---- a random undirected network with weights ----
W = nx.gnm_random_graph(25, 70, seed=11)
wr = random.Random(11)
for a, b in W.edges():
    W[a][b]["weight"] = round(wr.uniform(0.5, 4.0), 3)
write_network("weighted", W, True)
print("vector", "weighted", "constraint", fmt(per_node(W, nx.constraint(W, weight="weight"))))
print("vector", "weighted", "effective_size", fmt(per_node(W, nx.effective_size(W, nodes=list(W), weight="weight"))))
print("scalar", "weighted", "degree_assortativity_strength", repr(nx.degree_assortativity_coefficient(W, weight="weight")))
