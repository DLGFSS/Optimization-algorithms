# Errores al ejecutar un metodo de un objeto activo en AXO del algoritmo NSGA-ll
## 📘 Descripción
Se desarrolla un objeto activo en la plataforma AXO el objeto se registra correctamente, pero al intentar ejecutar el método run() o cualquier otro metodo se genera el siguiente error:
```
Err(JSONDecodeError('Expecting value: line 1 column 1 (char 0)'))
```
## Clase NSGA-ll 
```python
class NSGA2(Axo):
    
    def __init__(self): 
        self.by_row = 1
        
    def set_params(self, params):
        self.name = params["name"]
        self.label = params["label"]
        self.runs = params["runs"]
        self.iters = params["iters"]
        self.m_objs = params["m_objs"]
        self.pop_size = params["pop_size"]
        self.params_mop = params["params_mop"]
        self.params_crossover = params["params_crossover"]
        self.params_mutation = params["params_mutation"]
        self.verbose = params["verbose"]
        
    @axo_method
    def create_initial_population(self):
        return np.random.rand(self.pop_size, 10)  
    
    def dtlz1_g(self,Z, k):
        r = 100 * (k+(((Z - 0.5)**2) - np.cos(20*np.pi*(Z-0.5))).sum(axis=self.by_row))
        return r
    
    def dtlz1(self,X, m, M):
        r = None
        n = X.shape[1]
        j = M - 1
        k = n - j
        Y = X[:, :-k]
        Z = X[:, -k:]
        g = self.dtlz1_g(Z, k)
        if m == 0:
            r = (1.0 + g) * 0.5 * (Y.prod(axis=self.by_row))

        if m in range(1, M-1):
            r = (1.0 + g) * 0.5 * (Y[:, :M-m-1].prod(axis=self.by_row)) * (1 - Y[:, M-m-1])
    
        if m == M-1:
            r = (1.0 + g) * 0.5 * (1 - Y[:, 0])

        return r
    
    @axo_method
    def evaluate(self, population):
        X = np.array(population)
        objectives = np.array([self.dtlz1(X,m, self.m_objs) for m in range(self.m_objs)]).T
        return objectives

    @axo_method
    def select_parents(self, population, objectives):
        mating_pool = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, size=2, replace=False)
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                mating_pool.append(i)
            else:
                mating_pool.append(j)
        return mating_pool
    
    @axo_method
    def generate_new_population(self, population, mating_pool, objectives):
        new_population = np.copy(population)
        
        for i in range(0, self.pop_size, 2):
            parents = (population[mating_pool[i]], population[mating_pool[i+1]])
            child1, child2 = NSGA2.crossover(parents)
            new_population[i] = NSGA2.mutation(child1)
            new_population[i+1] = NSGA2.mutation(child2)
        
        return new_population
    
    @staticmethod
    def crossover(parents, prob=1.0, eta=20):
        if np.random.rand() < prob:
            c1, c2 = parents
            u = np.random.rand(*c1.shape)
            beta = np.ones_like(u)
            for i in range(len(u)):
                if u[i] <= 0.5:
                    beta[i] = (2 * u[i]) ** (1 / (eta + 1))
                else:
                    beta[i] = (1 / (2 * (1 - u[i]))) ** (1 / (eta + 1))
        
            child1 = 0.5 * ((1 + beta) * c1 + (1 - beta) * c2)
            child2 = 0.5 * ((1 - beta) * c1 + (1 + beta) * c2)
            return child1, child2
        else:
            return parents

    @staticmethod
    def mutation(child, prob=1.0, eta=15):
        if np.random.rand() < prob:
            u = np.random.rand(*child.shape)
            for i in range(len(child)):
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1 / (eta + 1))
                child[i] += delta
        return child

    @axo_method
    def run(self) -> dict:
        population = self.create_initial_population()
        objectives = self.evaluate(population)

        for gen in range(self.iters):
            if self.verbose:
                print(f"Generación {gen+1} completada")

            mating_pool = self.select_parents(population, objectives)
            population = self.generate_new_population(population, mating_pool, objectives)
            objectives = self.evaluate(population)
        result = {"population": population, "objectives": objectives}
        return result  
```
## Registro de objeto activo
```Python
from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2 , SumaPonderada
import logging

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
ax = ActiveXContextManager.distributed(endpoint_manager = endpoint_manager)

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

print(result)

```
## Resultado del registro del objeto activo en AXO
```
PS C:\Users\ddss2\Desktop\Educacion_dual\Moea_nuevo\nsga2_project> poetry run python Algoritmos.py   
2025-06-27 19:08:00,359 - activex.endpoint.endpoint - DEBUG - Connecting to tcp://localhost:16667
2025-06-27 19:08:00,364 - activex.endpoint.endpoint - DEBUG - Connected to tcp://localhost:16667
2025-06-27 19:08:00,364 - activex.endpoint.endpoint - DEBUG - {'event': 'PING', 'last_ping_at': -1, 'diff': 1751072881.3645093, 'max_health_tick_time': 3600.0, 'reqres_socket': '
<zmq.Socket(zmq.REQ) at 0x186e4100100>'}                                                                                                                                          2025-06-27 19:08:01,365 - activex.endpoint.endpoint - DEBUG - BEFORE.SEND.MULTIPLART
2025-06-27 19:08:01,365 - activex.endpoint.endpoint - DEBUG - BEGORE.SEND.AFTER
2025-06-27 19:08:01,370 - activex.endpoint.endpoint - DEBUG - ActiveX metadata service connected successfully.. tcp://*:16001
2025-06-27 19:08:02,134 - activex.endpoint.endpoint - DEBUG - PUT.METADATA wj7ygam6nrsts4nx 0.759864330291748
{
    "timestamp": "2025-06-27 19:08:02,171",
    "level": "INFO",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "event": "PUT.METADATA",
    "bucket_id": "0pqmc9ak2vzn9riwwh86fo4aevbylot2",
    "key": "wj7ygam6nrsts4nx_class_def",
    "replicas": [
        "mictlanx-peer-1"
    ],
    "service_time": 0.010487794876098633,
    "reponse_time": 0.03555703163146973
}

{
    "timestamp": "2025-06-27 19:08:02,240",
    "level": "INFO",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "event": "PUT.CHUNKED",
    "bucket_id": "0pqmc9ak2vzn9riwwh86fo4aevbylot2",
    "key": "wj7ygam6nrsts4nx_class_def",
    "checksum": "d7c6f715f8df1de345f93855720e25518c0e193cf5717affc91fcc3984fda168",
    "size": 65,
    "metadata_service_time": 0.010487794876098633,
    "metadata_response_time": 0.03555703163146973,
    "replicas": [
        "mictlanx-peer-1"
    ],
    "response_time": 0.10483527183532715
}

{
    "timestamp": "2025-06-27 19:08:02,284",
    "level": "INFO",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "event": "PUT.METADATA",
    "bucket_id": "0pqmc9ak2vzn9riwwh86fo4aevbylot2",
    "key": "wj7ygam6nrsts4nx",
    "replicas": [
        "mictlanx-peer-1"
    ],
    "service_time": 0.008437156677246094,
    "reponse_time": 0.03260183334350586
}

{
    "timestamp": "2025-06-27 19:08:02,327",
    "level": "INFO",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "event": "PUT.CHUNKED",
    "bucket_id": "0pqmc9ak2vzn9riwwh86fo4aevbylot2",
    "key": "wj7ygam6nrsts4nx",
    "checksum": "e944014d5a395f70550afb47fc9833b147777bac7afdae8dc82e64a844cca7ac",
    "size": 9026,
    "metadata_service_time": 0.008437156677246094,
    "metadata_response_time": 0.03260183334350586,
    "replicas": [
        "mictlanx-peer-1"
    ],
    "response_time": 0.07609128952026367
}

2025-06-27 19:08:02,328 - activex.v1 - INFO - {'event': 'PERSISTIFY', 'axo_bucket_id': '0pqmc9ak2vzn9riwwh86fo4aevbylot2', 'axo_key': 'wj7ygam6nrsts4nx', 'source_bucket_id': 'ztg
qajp1gwc4cvjvezyo0kl2qafu56e2', 'sink_bucket_id': 'oom1ntx9u4mglfvejuttf9q7jkimorqv', 'response_time': 0.9537642002105713}                                                        Ok('wj7ygam6nrsts4nx')
2025-06-27 19:08:02,362 - activex.runtime.distributed1 - DEBUG - Stop distributed runtime
```
## Utilizacion del metodo del algoritmo NSGA-ll registrado en AXO


```python
from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2 , SumaPonderada
import logging

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

def test():
    endpoint_manager = XoloEndpointManager()
    endpoint_manager.add_endpoint(
    endpoint_id  = AXO_ENDPOINT_ID,
    protocol     = AXO_ENDPOINT_PROTOCOL,
    hostname     = AXO_ENDPOINT_HOSTNAME,
    req_res_port = AXO_ENDPOINT_REQ_RES_PORT,
    pubsub_port  = AXO_ENDPOINT_PUBSUB_PORT)
    ax = ActiveXContextManager.distributed(endpoint_manager = endpoint_manager)
    key = "vqjlzb4po3n946q1"
    bucket_id = "92buxmi6d3t043tuqxrutwxux6ohrhg2"
    
    get_obj_result = Axo.get_by_key(bucket_id = bucket_id, key = key)
    
    if get_obj_result.is_ok:
            obj:NSGA2 = get_obj_result.unwrap()

            logger.debug({
                "event":"GET.BY.KEY",
                "bucket_id":bucket_id,
                "key":key,
                "obj":str(obj)
            })
            res        = obj.run()
            logger.debug({
                "method":"run",
                "result":res
            })
    else:
            logger.error({
                "msg":str(get_obj_result.unwrap_err())
            })

if __name__ == "__main__":
    test()


```
## Resultado de la utilizacion del metodo run de la clase NSGA2
```
PS C:\Users\ddss2\Desktop\Educacion_dual\Moea_nuevo\nsga2_project> poetry run python get_algoritms.py
2025-06-27 19:04:08,082 - activex.endpoint.endpoint - DEBUG - Connecting to tcp://localhost:16667
2025-06-27 19:04:08,089 - activex.endpoint.endpoint - DEBUG - Connected to tcp://localhost:16667
2025-06-27 19:04:08,089 - activex.endpoint.endpoint - DEBUG - {'event': 'PING', 'last_ping_at': -1, 'diff': 1751072649.0895467, 'max_health_tick_time': 3600.0, 'reqres_socket': '
<zmq.Socket(zmq.REQ) at 0x1da884701c0>'}                                                                                                                                          2025-06-27 19:04:09,103 - activex.endpoint.endpoint - DEBUG - BEFORE.SEND.MULTIPLART
2025-06-27 19:04:09,103 - activex.endpoint.endpoint - DEBUG - BEGORE.SEND.AFTER
2025-06-27 19:04:09,109 - activex.endpoint.endpoint - DEBUG - ActiveX metadata service connected successfully.. tcp://*:16001
{
    "timestamp": "2025-06-27 19:04:09,156",
    "level": "INFO",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "event": "GET",
    "bucket_id": "92buxmi6d3t043tuqxrutwxux6ohrhg2",
    "key": "vqjlzb4po3n946q1",
    "size": 9026,
    "response_time": 0.04249906539916992,
    "metadata_service_time": 0.019774436950683594,
    "peer_id": "mictlanx-peer-0"
}

2025-06-27 19:04:09,158 - __main__ - DEBUG - {'event': 'GET.BY.KEY', 'bucket_id': '92buxmi6d3t043tuqxrutwxux6ohrhg2', 'key': 'vqjlzb4po3n946q1', 'obj': '<common.definitios.NSGA2 
object at 0x000001DAA0E86080>'}                                                                                                                                                   2025-06-27 19:04:09,920 - activex.endpoint.endpoint - DEBUG - PUT.METADATA vqjlzb4po3n946q1 0.7614927291870117
{
    "timestamp": "2025-06-27 19:04:09,948",
    "level": "ERROR",
    "logger_name": "activex-mictlanx",
    "thread_name": "MainThread",
    "msg": [
        "{\"detail\":\"92buxmi6d3t043tuqxrutwxux6ohrhg2/vqjlzb4po3n946q1_class_def already exists.\"}"
    ],
    "status_code": 409
}

2025-06-27 19:04:09,950 - activex.runtime.runtime - ERROR - {'event': 'PERSISTIFY.FAILED', 'msg': '409 Client Error: Conflict for url: http://localhost:60666/api/v4/buckets/92bux
mi6d3t043tuqxrutwxux6ohrhg2/metadata'}                                                                                                                                            2025-06-27 19:04:09,950 - activex.v1 - ERROR - {'event': 'PERSISTIFY.FAILED', 'reason': '409 Client Error: Conflict for url: http://localhost:60666/api/v4/buckets/92buxmi6d3t043t
uqxrutwxux6ohrhg2/metadata'}                                                                                                                                                      2025-06-27 19:04:09,950 - activex.v1 - DEBUG - {'event': 'METHOD.EXECUTION', 'remote': False, 'local': True, 'fname': 'run', 'endpoint_id': 'activex-endpoint-0', 'axo_key': 'vqjl
zb4po3n946q1', 'axo_bucket_id': '92buxmi6d3t043tuqxrutwxux6ohrhg2', 'sink_bucket_id': '9jutvwiiulz72r7jh5ekwc43iz1biur0', 'source_bucket_id': 'mofc3j8ep24werzsui2f3dafrlxuaq8d'} 2025-06-27 19:04:10,050 - activex.v1 - INFO - {'event': 'METHOD.EXECUTION', 'remote': False, 'local': True, 'fname': 'run', 'endpoint_id': 'activex-endpoint-0', 'axo_key': 'vqjlz
b4po3n946q1', 'axo_bucket_id': '92buxmi6d3t043tuqxrutwxux6ohrhg2', 'sink_bucket_id': '9jutvwiiulz72r7jh5ekwc43iz1biur0', 'source_bucket_id': 'mofc3j8ep24werzsui2f3dafrlxuaq8d', 'response_time': 0.8914291858673096}                                                                                                                                               2025-06-27 19:04:10,050 - __main__ - DEBUG - {'method': 'run', 'result': Err(JSONDecodeError('Expecting value: line 1 column 1 (char 0)'))}
2025-06-27 19:04:10,080 - activex.runtime.distributed1 - DEBUG - Stop distributed runtime



```