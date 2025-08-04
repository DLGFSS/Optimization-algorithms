import networkx as nx
import matplotlib.pyplot as plt
import os

class BellmanFordAlgorithm:
    def __init__(self, graph):
        self.graph = graph

    def run(self, source, target):
        self.path, self.cost = nx.single_source_bellman_ford(self.graph, source=source)
        return self.path[target], self.cost[target]

    def draw_path(self, path, algorithm_name="BellmanFord"):
        os.makedirs("img", exist_ok=True)
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 5))
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightgreen')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='purple', width=3)

        plt.title(f"Camino más corto con {algorithm_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"img/{algorithm_name.lower()}_camino.png")
        plt.show()
