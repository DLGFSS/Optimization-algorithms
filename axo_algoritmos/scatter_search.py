from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import ScatterSearch
import logging
import time
import csv

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

  
def objective(x):
     return x**2

params = {
        "pop_size": 20,
        "refset_size": 5,
        "max_iter": 50
    }

ss = ScatterSearch(objective, lower=-100, upper=100, params=params)
mejor_x = ss.search()
print("Mejor solución encontrada:", mejor_x)
print("Valor de f(x):", objective(mejor_x))

res = ss.persistify()
print("Objeto activo: ", res )
