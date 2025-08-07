from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import SumaPonderada
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


n=32 
pasos = 50
TIEMPO =  []
    
for i in  range (n):
    print("ejecucion: ", i)
    inicio = time.time()    
    sp = SumaPonderada()
    final = time.time()
        
    duracion = final - inicio 
        
    TIEMPO.append((i+1 ,duracion))
    print(f"Tiempo de ejecución: {duracion:.8f} ")
        

    with open('tiempos_ejecucion.csv', mode='w', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        escritor.writerow(['Ejecucion', 'Tiempo (segundos)']) 
        escritor.writerows(TIEMPO)

    print("\nTiempos guardados en 'tiempos_ejecucion.csv'")
    
    

    soluciones, _ = sp.generar_frente_pareto(pasos=50)
    sp.graficar_frente_pareto(soluciones)
