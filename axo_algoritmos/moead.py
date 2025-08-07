from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2 , SumaPonderada ,MOEAD ,Problems
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
ax = ActiveXContextManager.distributed(endpoint_manager=endpoint_manager)



problem = Problems()

problem = problem.evaluate_zdt1


params = {
    "problem_func": problem,
    "n_var": 10,
    "bounds": [(0, 1)] * 10,
    "n_gen": 100,
    "n_sub": 100,
    "T": 20
}

        
moead = MOEAD()    
moead.set_params(params=params)
moead.evolve()
pareto = moead.get_pareto_front()
print("problema: ", problem)
f1_vals = [f[0] for f in pareto]
f2_vals = [f[1] for f in pareto]
#print("Pareto front:", pareto)
print("Cantidad de soluciones:", len(pareto))

plt.scatter(f1_vals, f2_vals ,s=10)
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("Pareto Front")
plt.savefig("pareto_front.png", dpi=300)
resultado = moead.persistify()  
print("objeto persistente: " , resultado)
        
      