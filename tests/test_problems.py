import numpy as np
import os
import sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.problems.problem import Problems

def eval_dtlz_fn_matrix(fn_id, chroms, m_objs):
    objs = np.zeros((chroms.shape[0], m_objs))
    if fn_id == 1:
        for m in range(m_objs):
            objs[:, m] = Problems.dtlz1(chroms, m, m_objs)
    return objs.copy()

def test_eval_dtlz1_shape():
    fn_id = 1
    m_objs = 3
    pop_size = 100
    n_genes = 30

    chroms = np.ones((pop_size, n_genes)) * 0.5
    chroms[:, 0] = np.linspace(0, 1, pop_size)
    chroms[:, 1] = np.linspace(0, 1, pop_size)

    objs = eval_dtlz_fn_matrix(fn_id, chroms, m_objs)

    # Verifica que la salida tenga la forma correcta
    assert objs.shape == (pop_size, m_objs)
    # Opcional: asegúrate de que todos los valores estén dentro de un rango razonable
    assert np.all(objs >= 0)
