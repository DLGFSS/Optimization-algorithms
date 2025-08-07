from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2 , SumaPonderada ,MOEAD ,Problems
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
ax = ActiveXContextManager.distributed(endpoint_manager=endpoint_manager)


problem = Problems()

problem = problem.dtlz1    
      
params = {
        "name": "nsga2",
        "label": "NSGA-II",
        "runs": 2, 
        "iters": 50,
        "m_objs": 2,
        "pop_size": 100,
        "problem_func": problem,
        "params_crossover": {"name": "sbx", "prob": 1.0, "eta": 20},
        "params_mutation": {"name": "polymutation", "prob": 1./6, "eta": 15},
        "verbose": True,
    }


nsga = NSGA2()
print("parametros: ",params )
nsga.set_params(params=params)
res = nsga.run()
nsga.set_dependencies(["numpy"])
logger.debug({
"result": res,
})    
    
result = nsga.persistify()
print("Objeto persistente: ",result)
        
        