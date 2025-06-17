import os
import sys
import time
import csv
import matplotlib.pyplot as plt

# Añadir la ruta al módulo MOEAD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from model.core.moead import MOEAD

def evaluate_zdt1(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - (f1 / g) ** 0.5)
    return [f1, f2]

# Parámetros del problema
n_var = 30
bounds = [(0, 1)] * n_var

# Número de ejecuciones
n_runs = 32
tiempos = []

# Ejecutar múltiples veces y medir tiempos
for i in range(n_runs):
    print(f"Ejecutando corrida {i + 1} de {n_runs}")
    inicio = time.time()

    moead = MOEAD(evaluate_zdt1, n_var, bounds, n_gen=200, n_sub=100, T=20)
    moead.evolve()

    fin = time.time()
    duracion = fin - inicio
    tiempos.append((i + 1, duracion))
    print(f"Tiempo de ejecución: {duracion:.6f} segundos")

# Guardar tiempos en CSV
csv_path = "tiempos_moead.csv"
with open(csv_path, mode='w', newline='') as archivo_csv:
    escritor = csv.writer(archivo_csv)
    escritor.writerow(['Ejecución', 'Tiempo (segundos)'])
    escritor.writerows(tiempos)

print(f"\nTiempos guardados en '{csv_path}'")

# Visualizar resultados de la última ejecución
pareto = moead.get_pareto_front()
f1_vals = [f[0] for f in pareto]
f2_vals = [f[1] for f in pareto]

plt.scatter(f1_vals, f2_vals, s=10)
plt.xlabel("f1")
plt.ylabel("f2")
plt.title("Aproximación al Frente de Pareto (ZDT1)")
plt.grid(True)
plt.show()
