# OptiKit

**OptiKit** is a modular and extensible Python toolkit for experimenting with and applying optimization algorithms — both classical and metaheuristic. It includes support for combinatorial, continuous, and multi-objective optimization problems. Designed for research, prototyping, and educational purposes.

---

## 🚀 Features

- ✅ Classical graph algorithms (e.g., Dijkstra, Bellman-Ford)
- 🔍 Local search and metaheuristics (e.g., Simulated Annealing, Tabu Search, Scatter Search)
- 🐝 Nature-inspired methods (e.g., Bees Algorithm, Cuckoo Search)
- 📈 Multi-objective evolutionary algorithms (NSGA-II, MOEA/D)
- 🔄 Unified interface for optimization problems
- 🧪 Built-in testing and benchmarking utilities

---

## 🧱 Project Structure
```bash
optikit/
├── algorithms/ # Core optimization algorithms
│ ├── bees_algorithm.py
│ ├── cuckoo_search.py
│ ├── simulated_annealing.py
│ ├── nsga2.py
│ └── ...
├── app/ # Application logic / workflows
│ ├── MOEA.py
│ ├── MOEAD.py
│ └── evaluate.py
├── problems/ # Problem definitions and interfaces
│ └── problem.py
tests/ # Unit tests for algorithms and problems
pyproject.toml # Poetry configuration
requirements.txt # Pip requirements
```

## 📦 Installation

### Option 1: Using [Poetry](https://python-poetry.org/)
```bash
poetry install

To install from the prebuilt package:
```bash
poetry add axo==0.0.3a0
```

⚠️ please use the newest version of Axo you can check it  https://github.com/muyal-research-group/axo



## 🧪 Running Tests

``` bash
pytest tests/
```

### 📏 Code Coverage
To evaluate how much of the codebase is exercised by the test suite, we use ```coverage```

```sh
coverage run -m pytest tests/
coverage report -m
```

- ```coverage run``` wraps the pytest invocation to trace executed statements.

- ```coverage report -m``` generates a summary including missing lines and module

This allows us to:

- Identify untested logic or edge cases.

- Focus testing efforts on high-risk areas.

- Prevent regressions by maintaining coverage thresholds over time.

You can also generate an HTML report for visual inspection:

```sh
coverage html
```
Then open ```htmlcov/index.html``` in your browser to explore annotated source files with coverage information.

## 🧠 Usage Example

```python
import random
from optikit.algorithms.simulated_annealing import SimulatedAnnealing
from axo.contextmanager import AxoContextMananger

with AxoContextManager.local() as lcm:
    sa:SimulatedAnnealing = SimulatedAnnealing(
        solucion_inicial= random.uniform(-10,10),
        temperatura=100.0,
        temperatura_minima=.0001,
        factor_enfriamiento= 0.95,
        axo_endpoint_id = "axo-endpoint-0"
    )
    # persist the object
    _ = await sa.persistify()
    res = sa.enfriamiento()


```

⚠️ The distributed version is in working progress please wait for the update of the ```axo``` core. 


## 🔬 Supported Algorithms
| Category        | Algorithms                                                     |
| --------------- | -------------------------------------------------------------- |
| Classical       | Dijkstra, Bellman-Ford                                         |
| Metaheuristics  | Simulated Annealing, Tabu Search, Scatter Search, Local Search |
| Evolutionary    | NSGA-II, MOEA/D                                                |
| Nature-inspired | Bees Algorithm, Cuckoo Search                                  |
| Multi-objective | NSGA-II, MOEA/D, Weighted Sum                                  |


## ✍️ Contributing
We welcome contributions! To add a new algorithm:

1. Create a module under optikit/algorithms/

2. Implement a class or function following the problem interface

3. Add unit tests under tests/

4. Open a pull request with a clear description
