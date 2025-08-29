import pytest


class Local_Search():
    def __init__(self, x):
        self.x = x
        self.trayectoria = [] 
        self.fx = []     

    def obj_function(self, x):
        return x**2

    def generate_neighbors(self, x ):
        return [x - 1, x + 1]

    def conditional(self, x, **kwargs):
        return abs(x) == 0  
    
    def local(self):
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


@pytest.mark.benchmark(group="baseline_local_search")
def test_baseline_Local_search(benchmark):
    ls:Local_Search = Local_Search(x=50)
        
    result = benchmark.pedantic(ls.local, iterations=10, rounds=100)
    assert isinstance(result, int)  
