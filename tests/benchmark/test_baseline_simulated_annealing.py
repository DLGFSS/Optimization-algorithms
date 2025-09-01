import pytest
import random

class Simulated_Annealing():
    def __init__(self, solucion_inicial, temperatura, temperatura_minima, factor_enfriamiento):
        self.solucion_inicial = solucion_inicial
        self.temperatura = temperatura
        self.temperatura_minima = temperatura_minima
        self.factor_enfriamiento = factor_enfriamiento
        self.coste_actual = self.evaluar_coste(self.solucion_inicial)
        self.historial_costes = []  
        
        
    def generar_vecino(self, solucion_actual):
        import random
        return solucion_actual + random.uniform(-1, 1)

    def evaluar_coste(self, solucion):
        return solucion**2
    
    def simulated(self):
        import random
        import math
        self.iteracion = 0
        while self.temperatura > self.temperatura_minima:
            nueva_solucion = self.generar_vecino(self.solucion_inicial)
            nuevo_coste = self.evaluar_coste(nueva_solucion)
            delta_e = nuevo_coste - self.coste_actual

            if delta_e < 0 or random.random() < math.exp(-delta_e / self.temperatura):
                self.solucion_inicial = nueva_solucion
                self.coste_actual = nuevo_coste

            self.historial_costes.append(self.coste_actual)

            self.temperatura *= self.factor_enfriamiento
            self.iteracion += 1

        return self.solucion_inicial, self.coste_actual, self.iteracion



@pytest.mark.benchmark(group="baseline_simulated")
def test_baseline_simulated_annealing(benchmark):
    def run_sa():
        sa = Simulated_Annealing(
            solucion_inicial=random.uniform(-10, 10),
            temperatura=100.0,
            temperatura_minima=0.0001,
            factor_enfriamiento=0.95
        )
        solucion, coste, iteraciones = sa.simulated()
        return coste 

    result = benchmark.pedantic(run_sa, iterations=10, rounds=100)
    assert isinstance(result, float)  
    