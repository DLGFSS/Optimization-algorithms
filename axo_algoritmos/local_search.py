from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import LocalSearch
import logging
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

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

busqueda = LocalSearch(x=50)
resultado = busqueda.search()

    # Gráfica de convergencia
plt.figure(figsize=(8,4))
plt.plot(busqueda.fx, marker='o', color='teal')
plt.title("Convergencia de Búsqueda Local en $f(x) = x^2$")
plt.xlabel("Iteración")
plt.ylabel("f(x)")
plt.grid()
plt.tight_layout()
plt.savefig("img/convergencia_local_search.png")
plt.show()

    # Gráfica de ruta sobre f(x)

x = np.linspace(-55, 55, 400)
y = x ** 2

plt.figure(figsize=(8,4))
plt.plot(x, y, label="f(x) = x²", color='lightgray')
plt.plot(busqueda.trayectoria, busqueda.fx, marker='o', color='crimson', label="Ruta del algoritmo")
plt.title("Ruta de exploración en $f(x) = x^2$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("img/ruta_local_search.png")
plt.show()

res = busqueda.persistify()
print("Objeto activo:", res)
