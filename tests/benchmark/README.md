# 🧪 Benchmark Comparativo: Axo vs Baseline

Este repositorio contiene pruebas de rendimiento automatizadas para comparar algoritmos implementados en Axo contra sus versiones baseline. Se utiliza `pytest-benchmark` para medir tiempos de ejecución y generar reportes reproducibles.

---

## 📁 Estructura de carpetas

```bash
tests/
└── benchmark/
    ├── test_baseline_simulated_annealing.py
    ├── test_axo_simulated_annealing_local.py
    ├── test_baseline_local_search.py
    ├── test_axo_local_search.py
    └── axo_vs_baseline.py  
```
## ⚙️ Preparación

Antes de ejecutar los benchmarks, asegúrate de crear la carpeta de reportes:

```bash
mkdir -p tests/benchmark/reports/histograms
```

## 🚀 Ejecución de benchmarks
- Simulated Annealing
```bash
pytest tests/benchmark/test_baseline_simulated_annealing.py \
       tests/benchmark/test_axo_simulated_annealing_local.py \
       --benchmark-only \
       --benchmark-autosave \
       --benchmark-compare \
       --benchmark-histogram=tests/benchmark/reports/histograms/simulated_annealing.html \
       --benchmark-json=tests/benchmark/reports/simulated_annealing.json
```
- Local Search
```bash
pytest tests/benchmark/test_baseline_local_search.py \
       tests/benchmark/test_axo_local_search.py \
       --benchmark-only \
       --benchmark-autosave \
       --benchmark-compare \
       --benchmark-histogram=tests/benchmark/reports/histograms/local_search.html \
       --benchmark-json=tests/benchmark/reports/local_search.json
```
## 📊 Comparación de Overhead

Una vez generados los reportes .json, ejecuta el script comparativo para calcular el overhead entre Axo y Baseline:
```bash
poetry run python tests/benchmark/axo_vs_baseline.py
```
Este script genera una tabla en consola con los tiempos medios (µs) y el porcentaje de overhead por algoritmo.

## 📌 Notas

    Los histogramas generados en HTML permiten visualizar la distribución de tiempos por ejecución.

    Los reportes .json incluyen metadatos del sistema, commit y estadísticas detalladas.

    El script axo_vs_baseline.py puede extenderse para generar tablas Markdown o visualizaciones adicionales.
