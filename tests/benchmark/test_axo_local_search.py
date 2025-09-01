import pytest
from optikit.algorithms.local_search import LocalSearch
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import matplotlib.pyplot as plt
import asyncio
from axo.storage.services import InMemoryStorageService

@pytest.mark.benchmark(group="axo_local_search")
def test_axo_Local_search(benchmark):
    async def run_local():
        with AxoContextManager.local() as rt:
            ls = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
            result = await ls.persistify()
            assert result.is_ok
            res = ls.local()
            assert res.is_ok
            mejor_solucion = res.unwrap()
        return mejor_solucion
    def runner():
        return asyncio.run(run_local())
    result = benchmark.pedantic(runner, iterations=10, rounds=100)
    assert isinstance(result, (int, float))
    

@pytest.mark.benchmark(group="axo_local_search_in_memory")
def test_axo_Local_search_in_memory(benchmark):
    ss = InMemoryStorageService()
    async def run_local():
        with AxoContextManager.local(storage_service=ss) as rt:
            ls = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
            result = await ls.persistify()
            assert result.is_ok
            res = ls.local()
            assert res.is_ok
            mejor_solucion = res.unwrap()
        return mejor_solucion
    def runner():
        return asyncio.run(run_local())
    result = benchmark.pedantic(runner, iterations=10, rounds=100)
    assert isinstance(result, (int, float))

