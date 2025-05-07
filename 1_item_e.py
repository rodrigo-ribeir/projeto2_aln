import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt

m_fixo = 100
n_max = 20
n_valores = np.arange(1, n_max + 1)
condicionamentos = []

for n in n_valores:
    A, c = poly_ls(m_fixo, n)
    kappa = cond(A)
    condicionamentos.append(kappa)

plt.figure(figsize=(10, 6))
plt.plot(n_valores, condicionamentos, marker='o')
plt.xlabel('Grau do Polinômio (n)')
plt.ylabel('Número de Condição da Matriz A')
plt.title(f'Condicionamento da Matriz A para m={m_fixo} e n até {n_max}')
plt.yscale('log')
plt.grid(True)
plt.show()

print("\nValores do número de condição:")
for n, kappa in zip(n_valores, condicionamentos):
    print(f"n = {n}: Condicionamento = {kappa:.2e}")