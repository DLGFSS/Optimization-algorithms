from activex import Axo, axo_methods
import numpy.typing as npt
import numpy as np
from typing import Optional,Tuple
from typing_extensions import Annotated
from activex.storage.mictlanx import GetKey
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
import os
import networkx as nx
import matplotlib.pyplot as plt


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
        self.problem_func = params["problem_func"]
        self.params_crossover = params["params_crossover"]
        self.params_mutation = params["params_mutation"]
        self.verbose = params["verbose"]
        
   
    
    def create_initial_population(self):
        return np.random.rand(self.pop_size, 10)  
    

    
    
    
    #se evalua la poblacion con la funcion de evaluacion del problema, se utiliza la funcion dtlz1 para evaluar la poblacion
    #
 
    def evaluate(self, population,*args, **kwargs):
        X = np.array(population)
        objectives = np.array([self.problem_func(X,m, self.m_objs) for m in range(self.m_objs)]).T
        return objectives

    #se seleccionan los padres para la reproduccion, se eligen dos padres aleatoriamente y se comparan sus objetivos
    # si el padre i es mejor que el padre j se selecciona el padre i, de lo contrario se selecciona el padre j
    #se repite el proceso hasta completar la poblacion de padres
    
    def select_parents(self, population, objectives):
        mating_pool = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, size=2, replace=False)
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                mating_pool.append(i)
            else:
                mating_pool.append(j)
        return mating_pool
    
    #se genera una nueva poblacion a partir de la poblacion actual y los padres seleccionados, se aplica el operador de cruce y mutacion
    #se generan dos hijos a partir de los padres seleccionados, se aplica el operador de crossover y mutation a cada hijo  
    
    def generate_new_population(self, population, mating_pool, objectives):
        new_population = np.copy(population)
        
        for i in range(0, self.pop_size, 2):
            parents = (population[mating_pool[i]], population[mating_pool[i+1]])
            child1, child2 = NSGA2.crossover(parents)
            new_population[i] = NSGA2.mutation(child1)
            new_population[i+1] = NSGA2.mutation(child2)
        
        return new_population
    
    #se aplica el operador de cruce y mutacion a los padres seleccionados, se generan dos hijos a partir de los padres seleccionados, se aplica el operador de crossover y mutation a cada hijo
    
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

    #se aplica el operador de mutacion a los hijos generados, se aplica el operador de mutacion a cada hijo
    
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


    #ejecucin del algortimo 
   
    def run(self ,*args, **kwargs) -> dict:
        print("Ejecutando NSGA-II")
        for r in range(self.runs):
            if self.verbose:
                print(f" Ejecución {r+1} de {self.runs}")

            population = self.create_initial_population()
            objectives = self.evaluate(population)

            for gen in range(self.iters):
                if self.verbose:
                    print(f"  Generación {gen+1} completada")

                mating_pool = self.select_parents(population, objectives)
                population = self.generate_new_population(population, mating_pool, objectives)
                objectives = self.evaluate(population)

                result = {"population": population, "objectives": objectives}
        return result

class Problems:
    def __init__(self):
        self.by_row = 1
    
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
    
    @staticmethod
    def evaluate_zdt1(x):
        f1 = x[0]
        g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
        f2 = g * (1 - (f1 / g) ** 0.5)
        return [f1, f2]

class SumaPonderada(Axo):

    def __init__(self):
        pass
    # Función objetivo 1
    def f1(self, x):
        return x[0]**2 + x[1]**2
     # Función objetivo 2
    def f2(self, x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    # Función objetivo combinada con pesos
    def weighted_sum_objective(self, x, weights):
        return weights[0]*self.f1(x) + weights[1]*self.f2(x)
    
    
    
    @axo_method
    def weighted_sum_method(self, weights, x0=[0, 0]):
        result = minimize(self.weighted_sum_objective, x0, args=(weights,))
        return result.x, result.fun


    @axo_method
    def generar_frente_pareto(self, pasos=50):
        solutions = []
        weights_list = []

        for alpha in np.linspace(0, 1, pasos):
            w = [alpha, 1 - alpha]
            x_opt, _ = self.weighted_sum_method(w)
            f1_val = self.f1(x_opt)
            f2_val = self.f2(x_opt)
            solutions.append([f1_val, f2_val])
            weights_list.append(w)

        return np.array(solutions), weights_list


    @axo_method
    def graficar_frente_pareto(self, soluciones):
        plt.figure(figsize=(8, 6))
        plt.plot(soluciones[:, 0], soluciones[:, 1], 'bo-', label='Frente de Pareto (estimado)')
        plt.xlabel('f1(x)')
        plt.ylabel('f2(x)')
        plt.title('Frente de Pareto aproximado por suma ponderada')
        plt.grid(True)
        plt.legend()
        plt.show()
 
class Individual:
    def __init__(self, n_var, bounds):
        self.n_var = n_var
        self.bounds = bounds
        self.x = [random.uniform(*bounds[i]) for i in range(n_var)]
        self.f = None

    def evaluate(self, problem_func):
        self.f = problem_func(self.x)

    def copy(self):
        clone = Individual(self.n_var, self.bounds)
        clone.x = self.x[:]
        clone.f = self.f[:]
        return clone

class MOEAD(Axo):
    def __init__(self):
        self.problem_func = None
        self.n_var = None
        self.bounds = None
        self.n_gen = 100
        self.n_sub = 100
        self.T = 20
        self.weights = None
        self.neighbors = None
        self.population = None
        self.z = None

    def set_params(self, params):
        self.problem_func = params["problem_func"]
        self.n_var = params["n_var"]
        self.bounds = params["bounds"]
        self.n_gen = params.get("n_gen", 100)
        self.n_sub = params.get("n_sub", 100)
        self.T = params.get("T", 20)

        self.weights = self.init_weights(self.n_sub)
        self.neighbors = self.get_neighbors(self.weights, self.T)
        self.population = [Individual(self.n_var, self.bounds) for _ in range(self.n_sub)]

        for ind in self.population:
            ind.evaluate(self.problem_func)

        self.z = [min([ind.f[i] for ind in self.population]) for i in range(2)]

    def evolve(self,*args, **kwargs) -> dict:
        for gen in range(self.n_gen):
            for i in range(self.n_sub):
                P = self.neighbors[i]
                p1, p2 = random.sample(P, 2)
                child = self.recombine(self.population[p1], self.population[p2])
                child.evaluate(self.problem_func)

                # Actualiza z*
                self.z = [min(self.z[j], child.f[j]) for j in range(2)]

                # Reemplazo
                for j in P:
                    f1 = self.scalarizing_chebyshev(child.f, self.weights[j], self.z)
                    f2 = self.scalarizing_chebyshev(self.population[j].f, self.weights[j], self.z)
                    if f1 < f2:
                        self.population[j] = child.copy()

    def recombine(self, ind1, ind2):
        child = Individual(self.n_var, self.bounds)
        child.x = [(x + y) / 2 for x, y in zip(ind1.x, ind2.x)]
        return child

    def get_pareto_front(self,*args, **kwargs) -> dict:
        return [ind.f for ind in self.population]

    def init_weights(self, n_subproblems):
        weights = []
        for i in range(n_subproblems):
            w = i / (n_subproblems - 1) if n_subproblems > 1 else 0
            weights.append([w, 1 - w])
        return weights

    def get_neighbors(self, weights, T):
        neighbors = []
        for i, w in enumerate(weights):
            dists = [(j, self.euclidean_dist(w, weights[j])) for j in range(len(weights))]
            dists.sort(key=lambda x: x[1])
            neighbors.append([j for j, _ in dists[:T]])
        return neighbors

    def scalarizing_chebyshev(self, f, weight, z):
        return max(weight[i] * abs(f[i] - z[i]) for i in range(len(f)))

    def euclidean_dist(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

class SimulatedAnnealing:
    def __init__(self, solucion_inicial, temperatura_inicial, temperatura_minima, factor_enfriamiento):
        self.solucion_actual = solucion_inicial
        self.temperatura = temperatura_inicial
        self.temperatura_minima = temperatura_minima
        self.factor_enfriamiento = factor_enfriamiento
        self.coste_actual = self.evaluar_coste(self.solucion_actual)
        self.historial_costes = []  
        
        
    def generar_vecino(self, solucion_actual):
        return solucion_actual + random.uniform(-1, 1)

    def evaluar_coste(self, solucion):
        return solucion**2

    def enfriamiento(self):
        iteracion = 0
        while self.temperatura > self.temperatura_minima:
            nueva_solucion = self.generar_vecino(self.solucion_actual)
            nuevo_coste = self.evaluar_coste(nueva_solucion)
            delta_e = nuevo_coste - self.coste_actual

            if delta_e < 0 or random.random() < math.exp(-delta_e / self.temperatura):
                self.solucion_actual = nueva_solucion
                self.coste_actual = nuevo_coste

            self.historial_costes.append(self.coste_actual)

            self.temperatura *= self.factor_enfriamiento
            iteracion += 1

        return self.solucion_actual, self.coste_actual, iteracion

class BeesAlgorithm(Axo):
    def __init__(self, objective_function, lower_bound, upper_bound, params):
        self.f = objective_function
        self.lower = lower_bound
        self.upper = upper_bound
        self.n = params["n"]       # Exploradoras
        self.m = params["m"]       # Sitios seleccionados
        self.e = params["e"]       # Sitios élite
        self.nep = params["nep"]   # Reclutas en élite
        self.nsp = params["nsp"]   # Reclutas en otros sitios
        self.ngh = params["ngh"]   # Radio del vecindario
        self.max_iter = params["max_iter"]

    def _random_solution(self):
        return random.uniform(self.lower, self.upper)

    def _neighborhood(self, x):
        delta = random.uniform(-self.ngh, self.ngh)
        neighbor = x + delta
        return max(self.lower, min(neighbor, self.upper))

    def search(self):
        os.makedirs("img", exist_ok=True)

        population = [self._random_solution() for _ in range(self.n)]
        best_solution = min(population, key=self.f)

        history_fx = []
        history_elite = []
        history_sites = []
        history_scouts = []

        for it in range(self.max_iter):
            population.sort(key=self.f)
            new_population = []

            elite_sites = []
            selected_sites = []

            # Élite
            for i in range(self.e):
                patch_center = population[i]
                recruits = [self._neighborhood(patch_center) for _ in range(self.nep)]
                best_patch = min(recruits, key=self.f)
                new_population.append(best_patch)
                elite_sites.append(best_patch)

            # Resto de sitios
            for i in range(self.e, self.m):
                patch_center = population[i]
                recruits = [self._neighborhood(patch_center) for _ in range(self.nsp)]
                best_patch = min(recruits, key=self.f)
                new_population.append(best_patch)
                selected_sites.append(best_patch)

            # Exploradoras
            scouts = [self._random_solution() for _ in range(self.n - self.m)]
            new_population.extend(scouts)

            # Historial
            population = new_population
            best_candidate = min(population, key=self.f)
            if self.f(best_candidate) < self.f(best_solution):
                best_solution = best_candidate

            history_fx.append(self.f(best_solution))
            history_elite.append(elite_sites)
            history_sites.append(selected_sites)
            history_scouts.append(scouts)

            # Guardar imagen por iteración
            self._plot_iteration(it, elite_sites, selected_sites, scouts)

        # Gráfica de convergencia
        self._plot_convergence(history_fx)

        return best_solution

    def _plot_iteration(self, iter_num, elites, sites, scouts):
        x_curve = np.linspace(self.lower, self.upper, 400)
        y_curve = x_curve ** 2

        plt.figure(figsize=(8, 4))
        plt.plot(x_curve, y_curve, color='lightgray', label="f(x) = x²")
        plt.scatter(scouts, [self.f(x) for x in scouts], color='skyblue', label='Exploradoras')
        plt.scatter(sites, [self.f(x) for x in sites], color='orange', label='Sitios Seleccionados')
        plt.scatter(elites, [self.f(x) for x in elites], color='crimson', label='Élite')
        plt.title(f"Iteración {iter_num + 1}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"img/bees_iter_{iter_num+1:02d}.png")
        plt.close()

    def _plot_convergence(self, fx_history):
        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='green')
        plt.title("Convergencia del Bees Algorithm")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("img/bees_convergencia.png")
        plt.show()

class BellmanFordAlgorithm(Axo):
    def __init__(self, graph):
        self.graph = graph

    def run(self, source, target):
        self.path, self.cost = nx.single_source_bellman_ford(self.graph, source=source)
        return self.path[target], self.cost[target]

    def draw_path(self, path, algorithm_name="BellmanFord"):
        os.makedirs("img", exist_ok=True)
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 5))
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightgreen')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='purple', width=3)

        plt.title(f"Camino más corto con bellman_ford")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"img/{algorithm_name.lower()}_camino.png")
        plt.show()

class DijkstraAlgorithm(Axo):
    def __init__(self, graph):
        self.graph = graph

    def run(self, source, target):
        # Devuelve el camino y el costo total desde source hasta target
        path, cost = nx.single_source_dijkstra(self.graph, source=source, target=target)
        return path, cost  # orden correcto: primero la lista del camino, luego el costo

    def draw_path(self, path, algorithm_name="Dijkstra"):
        os.makedirs("img", exist_ok=True)

        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 5))

        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightblue')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='crimson', width=3)

        plt.title(f"Camino más corto con Dijkstra")
        plt.axis('off')
        try:
            plt.tight_layout()
        except Exception:
                pass

        plt.savefig(f"img/{algorithm_name.lower()}_camino.png")
        plt.show()

class TabuSearch1D(Axo):
    def __init__(self, obj_function, neighborhood_fn, initial_solution, max_iter=50, tabu_size=5):
        self.f = obj_function
        self.neighbors = neighborhood_fn
        self.x0 = initial_solution
        self.max_iter = max_iter
        self.tabu_size = tabu_size

    def run(self):
        current = self.x0
        best = current
        tabu = []
        history = [self.f(best)]

        for _ in range(self.max_iter):
            candidates = [s for s in self.neighbors(current) if s not in tabu]
            if not candidates:
                break
            candidate = min(candidates, key=self.f)
            if self.f(candidate) < self.f(best):
                best = candidate
            tabu.append(candidate)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
            current = candidate
            history.append(self.f(best))

        self.plot_convergence(history)
        return best, self.f(best)

    def plot_convergence(self, history):
        os.makedirs("img", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(history, marker='o', color='darkorange')
        plt.title("Convergencia de Tabú Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("img/tabu_convergencia.png")
        plt.show()

class LocalSearch(Axo):
    def __init__(self, x):
        self.x = x
        self.trayectoria = []  
        self.fx = []     

    def obj_function(self, x):
        return x**2

    def generate_neighbors(self, x):
        return [x - 1, x + 1]

    def conditional(self, x):
        return abs(x) == 0  

    def search(self):
        s = self.x
        self.trayectoria.append(s)
        self.fx.append(self.obj_function(s))
        print("Punto de partida:", s)
        print("Valor objetivo inicial:", self.obj_function(s))
        
        while not self.conditional(s):
            neighbors = self.generate_neighbors(s)
            print("Vecinos:", neighbors)

            best_neighbor = None
            best_value = float('inf')

            for neighbor in neighbors:
                value = self.obj_function(neighbor)
                if value < best_value:
                    best_value = value
                    best_neighbor = neighbor

            if best_value < self.obj_function(s):
                s = best_neighbor
                self.trayectoria.append(s)
                self.fx.append(self.obj_function(s))
                
            else:
                break

        print("Mejor solución encontrada:", s)
        return s

class CuckooSearch(Axo):
    def __init__(self, objective, lower, upper, params):
        self.f = objective
        self.lower = lower
        self.upper = upper
        self.n = params["n"]              # Número de nidos
        self.pa = params["pa"]            # Probabilidad de descubrimiento
        self.max_iter = params["max_iter"]
        self.alpha = params.get("alpha", 1.0)  # Escala del vuelo Lévy

    def levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return self.alpha * step

    def simple_bounds(self, x):
        return max(self.lower, min(x, self.upper))

    def search(self):
        os.makedirs("img", exist_ok=True)

        nests = [random.uniform(self.lower, self.upper) for _ in range(self.n)]
        fitness = [self.f(x) for x in nests]
        best = nests[np.argmin(fitness)]
        fx_history = [self.f(best)]

        for _ in range(self.max_iter):
            for i in range(self.n):
                x = nests[i]
                step = self.levy_flight()
                new_x = self.simple_bounds(x + step)
                new_f = self.f(new_x)

                j = random.randint(0, self.n - 1)
                if new_f < fitness[j]:
                    nests[j] = new_x
                    fitness[j] = new_f

            for i in range(self.n):
                if random.random() < self.pa:
                    nests[i] = random.uniform(self.lower, self.upper)
                    fitness[i] = self.f(nests[i])

            current_best = nests[np.argmin(fitness)]
            if self.f(current_best) < self.f(best):
                best = current_best

            fx_history.append(self.f(best))

        self.plot_convergence(fx_history)
        return best

    def plot_convergence(self, fx_history):
        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='navy')
        plt.title("Convergencia de Cuckoo Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("img/cuckoo_convergencia.png")
        plt.show()
    
    
   
class ScatterSearch(Axo):
    def __init__(self, objective, lower, upper, params):
        self.f = objective
        self.lower = lower
        self.upper = upper
        self.pop_size = params["pop_size"]
        self.refset_size = params["refset_size"]
        self.max_iter = params["max_iter"]

    def initialize_population(self):
        return [random.uniform(self.lower, self.upper) for _ in range(self.pop_size)]

    def improve(self, solution):
        delta = random.uniform(-0.1, 0.1)
        candidate = solution + delta
        candidate = max(self.lower, min(self.upper, candidate))
        return candidate if self.f(candidate) < self.f(solution) else solution

    def improve_population(self, population):
        return [self.improve(sol) for sol in population]

    def update_refset(self, population):
        sorted_pop = sorted(population, key=self.f)
        return sorted_pop[:self.refset_size]

    def generate_subsets(self, refset):
        return [(refset[i], refset[j]) for i in range(len(refset)) for j in range(i + 1, len(refset))]

    def combine(self, s1, s2):
        return (s1 + s2) / 2

    def search(self):
        os.makedirs("img", exist_ok=True)

        pop = self.initialize_population()
        pop = self.improve_population(pop)
        refset = self.update_refset(pop)

        fx_history = [self.f(min(refset, key=self.f))]

        for _ in range(self.max_iter):
            subsets = self.generate_subsets(refset)
            new_solutions = []

            for s1, s2 in subsets:
                combined = self.combine(s1, s2)
                improved = self.improve(combined)
                new_solutions.append(improved)

            refset += new_solutions
            refset = self.update_refset(refset)

            best_fx = self.f(min(refset, key=self.f))
            fx_history.append(best_fx)

        self._plot_convergence(fx_history)

        return min(refset, key=self.f)

    def _plot_convergence(self, fx_history):
        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='purple')
        plt.title("Convergencia del Scatter Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("img/scatter_convergencia.png")
        plt.show()
  
class SimulatedAnnealing(Axo):
    def __init__(self, solucion_inicial, temperatura_inicial, temperatura_minima, factor_enfriamiento):
        self.solucion_actual = solucion_inicial
        self.temperatura = temperatura_inicial
        self.temperatura_minima = temperatura_minima
        self.factor_enfriamiento = factor_enfriamiento
        self.coste_actual = self.evaluar_coste(self.solucion_actual)
        self.historial_costes = []  
        
        
    def generar_vecino(self, solucion_actual):
        return solucion_actual + random.uniform(-1, 1)

    def evaluar_coste(self, solucion):
        return solucion**2

    def enfriamiento(self):
        iteracion = 0
        while self.temperatura > self.temperatura_minima:
            nueva_solucion = self.generar_vecino(self.solucion_actual)
            nuevo_coste = self.evaluar_coste(nueva_solucion)
            delta_e = nuevo_coste - self.coste_actual

            if delta_e < 0 or random.random() < math.exp(-delta_e / self.temperatura):
                self.solucion_actual = nueva_solucion
                self.coste_actual = nuevo_coste

            self.historial_costes.append(self.coste_actual)

            self.temperatura *= self.factor_enfriamiento
            iteracion += 1

        return self.solucion_actual, self.coste_actual, iteracion

