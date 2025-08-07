from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
import logging
import networkx as nx
from common.definitios import DijkstraAlgorithm, BellmanFordAlgorithm, TabuSearch1D


logger          = logging.getLogger(__name__)
formatter       = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

# Unique identifier of the endpoint
AXO_ENDPOINT_ID           = os.environ.get("AXO_ENDPOINT_ID","activex-endpoint-0")
AXO_ENDPOINT_PROTOCOL     = os.environ.get("AXO_ENDPOINT_PROTOCOL","tcp")
AXO_ENDPOINT_HOSTNAME     = os.environ.get("AXO_ENDPOINT_HOSTNAME","localhost")
AXO_ENDPOINT_PUBSUB_PORT  = int(os.environ.get("AXO_ENDPOINT_PUBSUB_PORT","16000"))
AXO_ENDPOINT_REQ_RES_PORT = int(os.environ.get("AXO_ENDPOINT_REQ_RES_PORT","16667"))


endpoint_manager = XoloEndpointManager()
endpoint_manager.add_endpoint(
endpoint_id  = AXO_ENDPOINT_ID,
protocol     = AXO_ENDPOINT_PROTOCOL,
hostname     = AXO_ENDPOINT_HOSTNAME,
req_res_port = AXO_ENDPOINT_REQ_RES_PORT,
pubsub_port  = AXO_ENDPOINT_PUBSUB_PORT)
ax = ActiveXContextManager.local()

edges = [
    ('A', 'B', 1),
    ('B', 'C', 2),
    ('C', 'A', 2),   
    ('C', 'Z', 2)
]



G = nx.DiGraph()
G.add_weighted_edges_from(edges)


# Dijkstra

dij = DijkstraAlgorithm(G)

cost, path = dij.run('A', 'Z')

print("Dijkstra:", path, "Coste:", cost)
dij.draw_path(path)

result_dij = dij.persistify()
print("Objeto activo Dijkstra:", result_dij)



# Bellman-Ford

bf = BellmanFordAlgorithm(G)

cost, path = bf.run('A', 'Z')
print("Bellman-Ford:", path, "Coste:", cost)
bf.draw_path(path)

result_bellman = bf.persistify()
print("Objeto activo Bellman-Ford:", result_bellman)


objective = lambda x: x ** 2
neighbors_fn = lambda x: [x - 1, x + 1]

tabu = TabuSearch1D(objective, neighbors_fn, initial_solution=25)
best, val = tabu.run()
print("Tabu Search:", best, "f(x):", val)

result_tabu = tabu.persistify()
print("Objeto activo Tabu Search:", result_tabu)