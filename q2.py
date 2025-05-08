# arquivo: minimos_quadrados.py

import numpy as np
from numpy.linalg import qr, svd, solve, pinv
import matplotlib.pyplot as plt

def poly_ls(t, n):
    """
    Gera a matriz de Vandermonde para ajuste polinomial.

    Args:
        t (numpy.ndarray): Vetor dos pontos independentes.
        n (int): Grau máximo do polinômio.

    Returns:
        numpy.ndarray: Matriz de Vandermonde de dimensões (len(t)) x (n+1).
    """
    m = len(t)
    A = np.zeros((m, n + 1))
    for j in range(n + 1):
        A[:, j] = t**j
    return A

def ls_qr(A, b):
    """
    Resolve o problema de mínimos quadrados Ax = b usando fatoração QR.

    Args:
        A (numpy.ndarray): Matriz A.
        b (numpy.ndarray): Vetor b.

    Returns:
        tuple: Vetor dos coeficientes x e o ajuste y = Ax.
    """
    Q, R = qr(A)
    x = solve(R, Q.T @ b)
    y = A @ x
    return x, y

def ls_svd(A, b):
    """
    Resolve o problema de mínimos quadrados Ax = b usando Decomposição em Valores Singulares (SVD).

    Args:
        A (numpy.ndarray): Matriz A.
        b (numpy.ndarray): Vetor b.

    Returns:
        tuple: Vetor dos coeficientes x e o ajuste y = Ax.
    """
    U, s, Vh = svd(A)
    Sigma_plus = np.zeros((A.shape[1], A.shape[0]))
    Sigma_plus[:A.shape[1], :A.shape[1]] = np.diag(1 / s)
    A_plus = Vh.T @ Sigma_plus @ U.T
    x = A_plus @ b
    y = A @ x
    return x, y

def ls_normal(A, b):
    """
    Resolve o problema de mínimos quadrados Ax = b usando o sistema de equações normais.

    Args:
        A (numpy.ndarray): Matriz A.
        b (numpy.ndarray): Vetor b.

    Returns:
        tuple: Vetor dos coeficientes x e o ajuste y = Ax.
    """
    x = solve(A.T @ A, A.T @ b)
    y = A @ x
    return x, y

if __name__ == '__main__':
    m = 100
    t = np.linspace(0, 1, m)

    # Funções a serem ajustadas
    f = np.sin(t)
    g = np.exp(t)
    h = np.cos(3 * t)

    # Grau do polinômio para regressão linear (n=1)
    n_linear = 1
    A_linear = poly_ls(t, n_linear)

    # Ajuste linear para seno
    x_qr_sin_lin, y_qr_sin_lin = ls_qr(A_linear, f)
    x_svd_sin_lin, y_svd_sin_lin = ls_svd(A_linear, f)
    x_normal_sin_lin, y_normal_sin_lin = ls_normal(A_linear, f)

    plt.figure(figsize=(8, 6))
    plt.plot(t, f, label='sin(t)')
    plt.plot(t, y_qr_sin_lin, '--', label='Ajuste Linear (QR)')
    plt.plot(t, y_svd_sin_lin, '--', label='Ajuste Linear (SVD)')
    plt.plot(t, y_normal_sin_lin, '--', label='Ajuste Linear (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Linear de sin(t)')
    plt.legend()
    plt.savefig('regressao_linear_sin.png')
    plt.close()

    # Ajuste linear para exponencial
    x_qr_exp_lin, y_qr_exp_lin = ls_qr(A_linear, g)
    x_svd_exp_lin, y_svd_exp_lin = ls_svd(A_linear, g)
    x_normal_exp_lin, y_normal_exp_lin = ls_normal(A_linear, g)

    plt.figure(figsize=(8, 6))
    plt.plot(t, g, label='exp(t)')
    plt.plot(t, y_qr_exp_lin, '--', label='Ajuste Linear (QR)')
    plt.plot(t, y_svd_exp_lin, '--', label='Ajuste Linear (SVD)')
    plt.plot(t, y_normal_exp_lin, '--', label='Ajuste Linear (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Linear de exp(t)')
    plt.legend()
    plt.savefig('regressao_linear_exp.png')
    plt.close()

    # Ajuste linear para cosseno
    x_qr_cos_lin, y_qr_cos_lin = ls_qr(A_linear, h)
    x_svd_cos_lin, y_svd_cos_lin = ls_svd(A_linear, h)
    x_normal_cos_lin, y_normal_cos_lin = ls_normal(A_linear, h)

    plt.figure(figsize=(8, 6))
    plt.plot(t, h, label='cos(3t)')
    plt.plot(t, y_qr_cos_lin, '--', label='Ajuste Linear (QR)')
    plt.plot(t, y_svd_cos_lin, '--', label='Ajuste Linear (SVD)')
    plt.plot(t, y_normal_cos_lin, '--', label='Ajuste Linear (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Linear de cos(3t)')
    plt.legend()
    plt.savefig('regressao_linear_cos.png')
    plt.close()

    # Grau do polinômio para ajuste de ordem 15 (n=15)
    n_poly15 = 15
    A_poly15 = poly_ls(t, n_poly15)

    # Ajuste polinomial de grau 15 para seno
    x_qr_sin_poly15, y_qr_sin_poly15 = ls_qr(A_poly15, f)
    x_svd_sin_poly15, y_svd_sin_poly15 = ls_svd(A_poly15, f)
    x_normal_sin_poly15, y_normal_sin_poly15 = ls_normal(A_poly15, f)

    plt.figure(figsize=(12, 8))
    plt.plot(t, f, label='sin(t)')
    plt.plot(t, y_qr_sin_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.plot(t, y_svd_sin_poly15, '--', label='Ajuste Polinomial (SVD)')
    plt.plot(t, y_normal_sin_poly15, '--', label='Ajuste Polinomial (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de sin(t)')
    plt.legend()
    plt.savefig('regressao_poly15_sin.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, f, label='sin(t)')
    plt.plot(t, y_qr_sin_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de sin(t) (QR)')
    plt.savefig('regressao_poly15_sin_qr.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, f, label='sin(t)')
    plt.plot(t, y_svd_sin_poly15, '--', label='Ajuste Polinomial (SVD)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de sin(t) (SVD)')
    plt.savefig('regressao_poly15_sin_svd.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, f, label='sin(t)')
    plt.plot(t, y_normal_sin_poly15, '--', label='Ajuste Polinomial (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de sin(t) (Normal)')
    plt.savefig('regressao_poly15_sin_normal.png')
    plt.close()

    # Ajuste polinomial de grau 15 para exponencial
    x_qr_exp_poly15, y_qr_exp_poly15 = ls_qr(A_poly15, g)
    x_svd_exp_poly15, y_svd_exp_poly15 = ls_svd(A_poly15, g)
    x_normal_exp_poly15, y_normal_exp_poly15 = ls_normal(A_poly15, g)

    plt.figure(figsize=(12, 8))
    plt.plot(t, g, label='exp(t)')
    plt.plot(t, y_qr_exp_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de exp(t)')
    plt.legend()
    plt.savefig('regressao_poly15_exp.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, g, label='exp(t)')
    plt.plot(t, y_qr_exp_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de exp(t) (QR)')
    plt.savefig('regressao_poly15_exp_qr.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, g, label='exp(t)')
    plt.plot(t, y_svd_exp_poly15, '--', label='Ajuste Polinomial (SVD)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de exp(t) (SVD)')
    plt.savefig('regressao_poly15_exp_svd.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, g, label='exp(t)')
    plt.plot(t, y_normal_exp_poly15, '--', label='Ajuste Polinomial (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de exp(t) (Normal)')
    plt.savefig('regressao_poly15_exp_normal.png')
    plt.close()

    # Ajuste polinomial de grau 15 para cosseno
    x_qr_cos_poly15, y_qr_cos_poly15 = ls_qr(A_poly15, h)
    x_svd_cos_poly15, y_svd_cos_poly15 = ls_svd(A_poly15, h)
    x_normal_cos_poly15, y_normal_cos_poly15 = ls_normal(A_poly15, h)

    plt.figure(figsize=(12, 8))
    plt.plot(t, h, label='cos(3t)')
    plt.plot(t, y_qr_cos_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de cos(3t)')
    plt.legend()
    plt.savefig('regressao_poly15_cos.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, h, label='cos(3t)')
    plt.plot(t, y_qr_cos_poly15, '--', label='Ajuste Polinomial (QR)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de cos(3t) (QR)')
    plt.savefig('regressao_poly15_cos_qr.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, h, label='cos(3t)')
    plt.plot(t, y_svd_cos_poly15, '--', label='Ajuste Polinomial (SVD)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de cos(3t) (SVD)')
    plt.savefig('regressao_poly15_cos_svd.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(t, h, label='cos(3t)')
    plt.plot(t, y_normal_cos_poly15, '--', label='Ajuste Polinomial (Normal)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Regressão Polinomial de Grau 15 de cos(3t) (Normal)')
    plt.savefig('regressao_poly15_cos_normal.png')
    plt.close()