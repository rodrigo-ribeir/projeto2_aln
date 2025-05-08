# questão 1 - item b

import numpy as np
from numpy.linalg import cond

m_values = [5, 10, 20, 50, 100, 1000, 10000, 100000]
for m in m_values:
    t = np.linspace(0, 1, m)
    A = np.array([[m, np.sum(t)], [np.sum(t), np.sum(t**2)]])
    condition_number = cond(A)
    print(f"Para m = {m}, o número de condição de A é: {condition_number:.2f}")