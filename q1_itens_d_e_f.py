import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt

def poly_ls(m, n):
    """
    Calcula a matriz A = X^T X para ajuste polinomial de grau até n.

    Args:
        m (int): Número de pontos de dados.
        n (int): Grau máximo do polinômio.

    Returns:
        numpy.ndarray: A matriz A de dimensões (n+1) x (n+1).
    """

    t = np.linspace(0, 1, m)
    A = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = np.sum(t**(i + j))

    return A


m_fixo = 100
n_max = 20
n_valores = np.arange(1, n_max + 1)
condicionamentos = []

for n in n_valores:
    A = poly_ls(m_fixo, n)
    kappa = cond(A)
    condicionamentos.append(kappa)

if __name__ == "__main___":
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

def poly_ls_centralizado(m, n):
    """
    Calcula a matriz A = X^T X para ajuste polinomial de grau até n com t centralizado.

    Args:
        m (int): Número de pontos de dados.
        n (int): Grau máximo do polinômio.

    Returns:
        numpy.ndarray: A matriz A de dimensões (n+1) x (n+1).
    """
    t = np.linspace(0, 1, m) - 1/2  # A principal modificação: centralização
    A = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = np.sum(t**(i + j))
    return A

if __name__ == '__main__':
    m_fixo = 100
    n_max = 20
    n_valores = np.arange(1, n_max + 1)
    condicionamentos_original = []
    condicionamentos_centralizado = []

    for n in n_valores:

        A_original = poly_ls(m_fixo, n)
        kappa_original = cond(A_original)
        condicionamentos_original.append(kappa_original)

        A_centralizado = poly_ls_centralizado(m_fixo, n)
        kappa_centralizado = cond(A_centralizado)
        condicionamentos_centralizado.append(kappa_centralizado)

    plt.figure(figsize=(12, 6))
    plt.plot(n_valores, condicionamentos_original, marker='o', label='t = i/m (Original)')
    plt.plot(n_valores, condicionamentos_centralizado, marker='x', label='t = i/m - 1/2 (Centralizado)')
    plt.xlabel('Grau do Polinômio (n)')
    plt.ylabel('Número de Condição da Matriz A')
    plt.title(f'Comparação do Condicionamento da Matriz A para m={m_fixo}')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparacao_condicionamento.png')
    plt.show()


    print("\nValores do número de condição (t = i/m):")
    for n, kappa in zip(n_valores, condicionamentos_original):
        print(f"n = {n}: Condicionamento = {kappa:.2e}")


    print("\nValores do número de condição (t = i/m - 1/2):")
    for n, kappa in zip(n_valores, condicionamentos_centralizado):
        print(f"n = {n}: Condicionamento = {kappa:.2e}")