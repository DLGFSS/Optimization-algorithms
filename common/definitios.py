from activex import Axo, axo_method

import numpy.typing as npt
import numpy as np
from typing import Optional,Tuple
from typing_extensions import Annotated
from activex.storage.mictlanx import GetKey
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class NSGA2(Axo):
    
    def __init__(self): 
        self.by_row = 1
        
    @axo_method
    def set_params(self, params):
        self.name = params["name"]
        self.label = params["label"]
        self.runs = params["runs"]
        self.iters = params["iters"]
        self.m_objs = params["m_objs"]
        self.pop_size = params["pop_size"]
        self.params_mop = params["params_mop"]
        self.params_crossover = params["params_crossover"]
        self.params_mutation = params["params_mutation"]
        self.verbose = params["verbose"]
        
    #se crea la poblacion aleatoriamente con las variables de decision 
    @axo_method
    def create_initial_population(self):
        return np.random.rand(self.pop_size, 10)  
    
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
    
    
    
    #se evalua la poblacion con la funcion de evaluacion del problema, se utiliza la funcion dtlz1 para evaluar la poblacion
    #
    @axo_method
    def evaluate(self, population):
        X = np.array(population)
        objectives = np.array([self.dtlz1(X,m, self.m_objs) for m in range(self.m_objs)]).T
        return objectives

    #se seleccionan los padres para la reproduccion, se eligen dos padres aleatoriamente y se comparan sus objetivos
    # si el padre i es mejor que el padre j se selecciona el padre i, de lo contrario se selecciona el padre j
    #se repite el proceso hasta completar la poblacion de padres
    @axo_method
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
    @axo_method
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
    @axo_method
    def run(self) -> dict:
        population = self.create_initial_population()
        objectives = self.evaluate(population)

        for gen in range(self.iters):
            if self.verbose:
                print(f"Generación {gen+1} completada")

            mating_pool = self.select_parents(population, objectives)
            population = self.generate_new_population(population, mating_pool, objectives)
            objectives = self.evaluate(population)

        # Empaquetar resultados en un diccionario antes de devolverlos
        result = {"population": population, "objectives": objectives}
    
        return result  # Evita problemas con el desempaquetado en el entorno activo



class SumaPonderada(Axo):
    
    def __init__(self):
        pass
#     # Función objetivo 1
    def f1(self, x):
        return x[0]**2 + x[1]**2
#     # Función objetivo 2
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
