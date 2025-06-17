from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2 , SumaPonderada

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
    pubsub_port  = AXO_ENDPOINT_PUBSUB_PORT
)
ax = ActiveXContextManager.distributed(
    endpoint_manager = endpoint_manager
)

params = {
        "name": "nsga2",
        "label": "NSGA-II",
        "runs": 2, 
        "iters": 50,
        "m_objs": 2,
        "pop_size": 100,
        "params_mop": {"name": "dtlz1"},
        "params_crossover": {"name": "sbx", "prob": 1.0, "eta": 20},
        "params_mutation": {"name": "polymutation", "prob": 1./6, "eta": 15},
        "verbose": True,
}


nsga = NSGA2()
nsga.set_params(params=params)
nsga.set_dependencies(["numpy"])
result = nsga.persistify()
print("NSGA2  Objeto Persistente con Key:", result)


#obj = Axo.get_by_key(key = "8exn35gsfoqmufh8",bucket_id = "yqjablbggesccngwzy5smw91kqt4y6b0")
#result = obj.unwrap()
#result = result.create_initial_population()
#print("Objeto NSGA2 Deserializado:", result)

suma = SumaPonderada()
suma.set_dependencies(["numpy", "scipy", "matplotlib"])
result = suma.persistify()
print("SumaPonderada Objeto Persistente con Key:", result)