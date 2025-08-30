import pytest
import random
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from optikit.algorithms.simulated_annealing import SimulatedAnnealing

    
    
@pytest.mark.benchmark(group="axo_simulated")
def test_axo_simulated_annealing(benchmark):
    def run_sa():
        with AxoContextManager.local():
         sa = SimulatedAnnealing(
            solucion_inicial=random.uniform(-10, 10),
            temperatura=100.0,
            temperatura_minima=0.0001,
            factor_enfriamiento=0.95,
            axo_endpoint_id = "axo-endpoint-0"
        )
        sa.persistify()
        res = sa.simulated()
        print(res)
        assert res.is_ok
        solucion, coste, iteraciones  = res.unwrap()
       
        return coste 
    result = benchmark.pedantic(run_sa, iterations=10, rounds=100)
    assert isinstance(result, float) 

