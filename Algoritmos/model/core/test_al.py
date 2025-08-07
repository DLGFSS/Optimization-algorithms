import networkx as nx

edges = [
    ('A', 'B', 1),
    ('B', 'C', -2),
    ('C', 'A', -2),   
    ('C', 'Z', 2)
]



G = nx.DiGraph()
G.add_weighted_edges_from(edges)


# Dijkstra

from dijkstra import DijkstraAlgorithm
dij = DijkstraAlgorithm(G)

cost, path = dij.run('A', 'Z')
print("Dijkstra:", path, "Coste:", cost)
dij.draw_path(path)

# Bellman-Ford

from axo_algoritmos.bellaman_ford import BellmanFordAlgorithm
bf = BellmanFordAlgorithm(G)

cost, path = bf.run('A', 'Z')
print("Bellman-Ford:", path, "Coste:", cost)
bf.draw_path(path)

from tabu_search import TabuSearch1D

objective = lambda x: x ** 2
neighbors_fn = lambda x: [x - 1, x + 1]

tabu = TabuSearch1D(objective, neighbors_fn, initial_solution=25)
best, val = tabu.run()
print("Tabu Search:", best, "f(x):", val)
