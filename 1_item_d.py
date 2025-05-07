import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt

def poly_ls(m, n, b):
    """
    Calcula a matriz A = X^T X e o vetor c = X^T b para ajuste polinomial de grau até n.

    Args:
        m (int): Número de pontos de dados.
        n (int): Grau máximo do polinômio.
        b (numpy.ndarray): Vetor dos valores observados (dimensão m).

    Returns:
        numpy.ndarray: A matriz A de dimensões (n+1) x (n+1).
        numpy.ndarray: O vetor c de dimensão (n+1).
    """

    t = np.linspace(0, 1, m)
    A = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = np.sum(t**(i + j))

    t = np.linspace(0, 1, m)
    c = np.zeros(n + 1)
    for i in range(n + 1):
        c[i] = np.sum(b * (t**i))
    return A, c


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