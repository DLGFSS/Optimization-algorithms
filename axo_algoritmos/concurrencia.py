from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import time
import os
import logging
import csv
from concurrent.futures import ThreadPoolExecutor
from common.definitios import MOEAD, Problems  


logger          = logging.getLogger(__name__)
formatter       = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)

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

num_ejecuciones = 2  
hilos = 5      
def ejecutar_moead(i):
    try:
        moead = MOEAD()
        moead.set_params(params=params)

        inicio_persistencia = time.time()
        resultado = moead.persistify()
        fin_persistencia = time.time()

        duracion_persistencia = fin_persistencia - inicio_persistencia

        
        return (i, duracion_persistencia)
    
    except Exception as e:
        return (i, None)

with ThreadPoolExecutor(hilos=hilos) as executor:
    resultados = list(executor.map(ejecutar_moead, range(1, num_ejecuciones + 1)))

with open("tiempo_almacenamiento.csv", mode="w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ejecucion", "Tiempo de almacenamiento"])
    for fila in resultados:
        writer.writerow(fila)

print("Archivo CSV guardado con solo los tiempos de almacenamiento.")
