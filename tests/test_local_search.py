import pytest
from optikit.algorithms.local_search import LocalSearch
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import matplotlib.pyplot as plt


@pytest.fixture()
def dem():
    dem = DistributedEndpointManager()
    dem.add_endpoint(
        endpoint_id  = "axo-endpoint-0",
        hostname     = "localhost",
        protocol     = "tcp",
        req_res_port = 16667,
        pubsub_port  = 16666
    )
    return dem

@pytest.mark.asyncio
async def test_local_Local_search():
    with AxoContextManager.local() as dcm:
        ls:LocalSearch = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
        _ = await ls.persistify()
        res = ls.local()
        assert res.is_ok
        mejor_solucion = res.unwrap()
        
        print("Mejor solución encontrada:", mejor_solucion)
        
        # Gráfica de convergencia

