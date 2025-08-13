from optikit.algorithms.bellaman_ford import BellmanFordAlgorithm
from optikit.algorithms.moead import MOEAD
from optikit.algorithms.nsga2 import NSGA2
from optikit.algorithms.suma_ponderada import SumaPonderada
from optikit.algorithms.tabu_search import TabuSearch1D
from optikit.algorithms.scatter_search import ScatterSearch
from optikit.algorithms.simulated_annealing import SimulatedAnnealing
from optikit.algorithms.dijkstra import DijkstraAlgorithm
from optikit.algorithms.local_search import LocalSearch
from optikit.algorithms.bees_algorithm import BeesAlgorithm
from optikit.algorithms.cuckoo_search import CuckooSearch
from optikit.problems.problem import Problems
import pytest
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import networkx as nx
import random
import matplotlib.pyplot as plt
import asyncio


def objective_function(x):
    return (x - 10) ** 2


def neighbors_fn(x):
    return [x - 1, x + 1]



edges = [
    ('A', 'B', 1),
    ('B', 'C', 2),
    ('C', 'A', 2),   
    ('C', 'Z', 2)
]

G = nx.DiGraph()
G.add_weighted_edges_from(edges)



problem = Problems()
fn_problem = problem.evaluate_zdt1

n_var = 50
bounds = [(0, 1)] * n_var
n_runs = 2
tiempos = []



           
           

async def nsga2():
    params = {
            "name": "nsga2",
            "label": "NSGA-II",
            "runs": 250, 
            "iters": 50,
            "m_objs": 2,
            "pop_size": 100,
            "params_mop": {"name": "dtlz1"},
            "params_crossover": {"name": "sbx", "prob": 1.0, "eta": 20},
            "params_mutation": {"name": "polymutation", "prob": 1./6, "eta": 15},
            "verbose": True,}
    
    with AxoContextManager.local() as dcm:
        nsga2: NSGA2 = NSGA2(params)
        nsga2.persistify()
        result = nsga2.nsga()
        assert result.is_ok
        population , objectives  = result.unwrap()
        print("Final Population:\n", population)
        print("Objectives:\n", objectives)
        
async def bellaman_ford():
    with AxoContextManager.local() as dcm:
        bf:BellmanFordAlgorithm = BellmanFordAlgorithm(G, axo_endpoint_id="axo-endpoint-0")
        _ = await bf.persistify()
        res = bf.bellaman('A', 'Z')
        assert res.is_ok
        print(res)
        # cost, path = res.unwrap()
        # print("Bellaman ford:", path, "Coste:", cost)
        # bf.draw_path(path)

async def bess_algorithm():
    with AxoContextManager.local() as dcm:
        ba: BeesAlgorithm =  BeesAlgorithm(
            objective_function=lambda x: x ** 2,
            lower_bound=-10,
            upper_bound=10,
            params={
                "n": 10,
                "m": 10,
                "e": 5,
                "nep": 3,
                "nsp": 2,
                "ngh": 0.5,
                "max_iter": 25
            },
            axo_endpoint_id="axo-endpoint-0"
        )
        _ = await ba.persistify()
        best_solution = ba.bees()
        assert best_solution.is_ok
        print("Best solution found:", best_solution.unwrap())
        
async def cuckoo_search():
    with AxoContextManager.local() as dcm:
        cs: CuckooSearch =  CuckooSearch(
            objective_function,
            lower=-10, 
            upper=10, 
            params = {
                "n": 50,
                "pa": 0.25,
                "max_iter": 100,
                "alpha": 1.0
        },
            axo_endpoint_id="axo-endpoint-0"
        )
        _ = await cs.persistify()
        res = cs.cuckoo()
        assert res.is_ok
        best = res.unwrap()
        print("Best solution found:", best)        
        
async def dijkstra():
    with AxoContextManager.local() as dcm:
        ad:DijkstraAlgorithm = DijkstraAlgorithm(G, axo_endpoint_id="axo-endpoint-0")
        _ = await ad.persistify()
        res = ad.dijkstra('A', 'Z')
        assert res.is_ok
        cost, path = res.unwrap()
        print("Dijkstra:", path, "Coste:", cost)
        
              
async def Local_search():
    with AxoContextManager.local() as dcm:
        ls:LocalSearch = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
        _ = await ls.persistify()
        res = ls.local()
        assert res.is_ok
        mejor_solucion = res.unwrap()
        
        print("Mejor solución encontrada:", mejor_solucion)
    

async def moead():
 
    with AxoContextManager.local() as dcm:
        moead: MOEAD = MOEAD(fn_problem, n_var, bounds, n_gen=200, n_sub=100, T=20)
        _ = await moead.persistify()
        result = moead.moea()
        assert result.is_ok
        pareto = moead.get_pareto_front()
        print("Pareto Front:", pareto)
        
async def scatter_search():
    with AxoContextManager.local() as dcm:
        sc: ScatterSearch =  ScatterSearch(
            objective_function,
            lower=-10, 
            upper=10, 
            params = {
                "pop_size": 50,
                "refset_size": 5,
                "max_iter": 50
            },
            axo_endpoint_id="axo-endpoint-0"
        )
        _ = await sc.persistify()
        mejor_x = sc.scatter()
        assert mejor_x.is_ok
        best = mejor_x.unwrap()
        print("Best solution found:", best)
        
async def simulated_annealing():
    with AxoContextManager.local() as lcm:
        sa:SimulatedAnnealing = SimulatedAnnealing(
            solucion_inicial= random.uniform(-10,10),
            temperatura=100.0,
            temperatura_minima=.0001,
            factor_enfriamiento= 0.95,
            axo_endpoint_id = "axo-endpoint-0"
        )
        _ = await sa.persistify() 
        res = sa.simulated()
        print(res)
        assert res.is_ok
        solucion, coste, iteraciones  = res.unwrap()
        
        print(f"Solución encontrada: x = {solucion:.5f}")
        print(f"Coste final: f(x) = {coste:.5f}")
        print(f"Iteraciones: {iteraciones}")

        
async def suma_ponderada():
    with AxoContextManager.local() as dcm:
        sp:SumaPonderada = SumaPonderada(
        
            axo_endpoint_id="axo-endpoint-0")
        _ = await sp.persistify()
        res = sp.suma(pasos=50)
        print("Frente de Pareto generado con éxito:", res)
        assert res.is_ok
        solutions = res.unwrap()
        sp.graficar_frente_pareto(solutions[0])
        
async def tabu_search():
    
    with AxoContextManager.local() as dcm:
        ts:TabuSearch1D = TabuSearch1D(
            objective_function,
            neighbors_fn, 
            initial_solution=25,
            axo_endpoint_id="axo-endpoint-0")
        _ = await ts.persistify()
        res = ts.tabu()
        assert res.is_ok
        best ,a = res.unwrap()
        print("Tabu Search:", best,a)
        


async_functions = [
    nsga2,
    bellaman_ford,
    bess_algorithm,
    cuckoo_search,
    dijkstra,
    Local_search,
    moead,
    scatter_search,
    simulated_annealing,
    suma_ponderada,
    tabu_search
]

async def run_function_n_times(func, n=50):
    for i in range(n):
        print(f"Ejecutando {func.__name__} iteración {i+1}/{n}")
        await func()

async def main():
    for func in async_functions:
        await run_function_n_times(func, 50)

if __name__ == "__main__":
    asyncio.run(main())

