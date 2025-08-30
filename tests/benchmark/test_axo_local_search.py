import pytest
from optikit.algorithms.local_search import LocalSearch
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import matplotlib.pyplot as plt

    
@pytest.mark.benchmark(group="axo_local_search")
def test_axo_Local_search(benchmark):
    with AxoContextManager.local():
        ls = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
        
        def run_local():
            ls.persistify()
            return ls.local().unwrap_or(0)

        result = benchmark.pedantic(run_local, iterations=10, rounds=100)

    assert isinstance(result, (int, float))


