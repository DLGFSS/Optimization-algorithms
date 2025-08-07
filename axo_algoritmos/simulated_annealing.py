import random
import math
from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import SimulatedAnnealing
import logging
import time
import csv
import matplotlib.pyplot as plt

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


sa = SimulatedAnnealing(
        solucion_inicial = random.uniform(-10, 10),
        temperatura_inicial = 100.0,
        temperatura_minima = 0.001,
        factor_enfriamiento = 0.95
    )

solucion, coste, iteraciones = sa.enfriamiento()

print(f"Solución encontrada: x = {solucion:.5f}")
print(f"Coste final: f(x) = {coste:.5f}")
print(f"Iteraciones: {iteraciones}")

plt.plot(sa.historial_costes)
plt.title("Evolución del coste en Simulated Annealing")
plt.xlabel("Iteración")
plt.ylabel("Coste f(x)")
plt.grid(True)
plt.show()
