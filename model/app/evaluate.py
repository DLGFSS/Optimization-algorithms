import numpy as np
from nsga2_project.model.problems.problem import Problems

def evaluate_population(population):
    return np.array([Problems.dtlz1(ind) for ind in population])
