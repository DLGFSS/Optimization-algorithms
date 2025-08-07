from activex import Axo
from activex.contextmanager.contextmanager import ActiveXContextManager
from activex.endpoint import XoloEndpointManager
import os
from common.definitios import NSGA2
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

bucket_id = "ve6tx38e5fg1ameyp5wnrhciwd7rs0y2"
key = "0vse3wa8sprjgc1r"

print("__________________________Traer el objeto activo__________________________", key)
get_obj = Axo.get_by_key(bucket_id=bucket_id,key=key)
print("get_obj: ", get_obj)

if get_obj.is_ok:
    obj:NSGA2 = get_obj.unwrap()
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
    print("Error al obtener el objeto activo:")
    logger.error({
                "msg":str(get_obj.unwrap_err())
            })
    
