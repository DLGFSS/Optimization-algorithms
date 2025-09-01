import pytest
import random
import asyncio
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from axo.storage.services import InMemoryStorageService
from optikit.algorithms.simulated_annealing import SimulatedAnnealing

    
    
@pytest.mark.benchmark(group="axo_simulated")
def test_axo_simulated_annealing(benchmark):
    async def run_sa():
        with AxoContextManager.local() as rt:
            sa = SimulatedAnnealing(
                solucion_inicial=random.uniform(-10, 10),
                temperatura=100.0,
                temperatura_minima=0.0001,
                factor_enfriamiento=0.95,
                axo_endpoint_id = "axo-endpoint-0"
            )
            persistify_result = await sa.persistify()
            assert persistify_result.is_ok
            res =  sa.simulated()
            assert res.is_ok
            solucion, coste, iteraciones  = res.unwrap()
       
        return coste 
    def runner():
       return asyncio.run(run_sa())
    result = benchmark.pedantic(runner, iterations=10, rounds=100)
    assert isinstance(result, float) 

@pytest.mark.benchmark(group="axo_simulated_in_memory")
def test_axo_simulated_annealing_in_memory(benchmark):
    ss = InMemoryStorageService()
    async def run_sa():
        with AxoContextManager.local(storage_service=ss) as rt:
            sa = SimulatedAnnealing(
                solucion_inicial=random.uniform(-10, 10),
                temperatura=100.0,
                temperatura_minima=0.0001,
                factor_enfriamiento=0.95,
                axo_endpoint_id = "axo-endpoint-0"
            )
            persistify_result = await sa.persistify()
            assert persistify_result.is_ok
            res =  sa.simulated()
            assert res.is_ok
            solucion, coste, iteraciones  = res.unwrap()
       
        return coste 
    def runner():
       return asyncio.run(run_sa())
    result = benchmark.pedantic(runner, iterations=10, rounds=100)
    assert isinstance(result, float) 
