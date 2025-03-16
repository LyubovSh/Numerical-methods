#Разработать и сравнить эффективность прямого метода циклической прогон-ки, итерационного метода Гаусса, метода Холецкого и метода наискорейшего спуска с трехдиагональной* системой разного размера. Сравнение будет про-водиться по таким критериям, как скорость сходимости, вычислительная сложность и точность полученного решения.
import numpy as np
import time
from random import randint


def get_alphas(A, f):
    res = []
    res.append(-A[1][2] / A[1][1])
    for i in range(2, len(f) + 1):
        res.append(-A[i % len(f)][(i + 1) % len(f)] / (A[i % len(f)][i % len(f)] + A[i % len(f)][i - 1] * res[-1]))
    return res


def get_betas(A, f, arr):
    res = []
    res.append(f[1] / A[1][1])
    for i in range(2, len(f) + 1):
        a = A[i % len(f)][i - 1]
        res.append((f[i % len(f)] - a * res[-1]) / (A[i % len(f)][i % len(f)] + a * arr[i - 2]))
    return res


def get_gammas(A, f, arr):
    res = []
    res.append(-A[1][0] / A[1][1])
    for i in range(2, len(f) + 1):
        a = A[i % len(f)][i - 1]
        res.append((-a * res[-1]) / (A[i % len(f)][i % len(f)] + a * arr[i - 2]))
    return res


def get_u(alphas, betas):
    length = len(alphas)
    ans = [0.0 for i in range(length - 1)]
    ans[length - 2] = betas[length - 2]
    for i in range(length - 3, -1, -1):
        ans[i] = ans[i + 1] * alphas[i] + betas[i]
    return ans


def get_v(alphas, gammas):
    length = len(alphas)
    ans = [0.0 for i in range(length - 1)]
    ans[length - 2] = gammas[length - 2] + alphas[length - 2]
    for i in range(length - 3, -1, -1):
        ans[i] = ans[i + 1] * alphas[i] + gammas[i]
    return ans


def cyclic_tridiagonal_solver(A, f):
    start = time.time()
    alphas = get_alphas(A, f)
    betas = get_betas(A, f, alphas)
    gammas = get_gammas(A, f, alphas)
    u = get_u(alphas, betas)
    v = get_v(alphas, gammas)
    y_0 = (betas[-1] + alphas[-1] * u[0]) / (1 - gammas[-1] - alphas[-1] * v[0])
    res = [y_0]
    for i in range(0, len(f) - 1):
        res.append(u[i] + v[i] * y_0)
    return res, time.time() - start




def gauss_elimination(matrix, vector):
    start = time.time()
    n = len(vector)
    augmented_matrix = np.hstack((matrix.astype(float), vec-tor.reshape(-1, 1)))

    for i in range(n):
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        if augmented_matrix[max_row, i] == 0:
            raise ValueError("Система не имеет уникального ре-шения.")

        augmented_matrix[[i, max_row]] = augment-ed_matrix[[max_row, i]]

        augmented_matrix[i] /= augmented_matrix[i, i]

        for j in range(i + 1, n):
            augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])

    finish = time.time()
    t = (finish - start)

    return x, t


def cholesky_solve(A, b):
    start = time.time()
    if not np.allclose(A, A.T):
        raise ValueError("Матрица A должна быть симметричной.")
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise ValueError("Матрица A не является положительно определённой.")
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L.T[i, i + 1:], x[i + 1:])) / L[i, i]

    return x, time.time() - start

def gradient_descent(A, b, x0, tol=1e-8, max_iter=10000000):
    start = time.time()
    x = x0
    r = A @ x - b
    iter_count = 0

    while np.linalg.norm(r) > tol and iter_count < max_iter:
        alpha = (r @ r) / (r @ (A @ r))
        x = x - alpha * r
        r = A @ x - b
        iter_count += 1

    if (iter_count == max_iter):
        return None
    return x, time.time() - start



size = 10
for i in range(3):
    size = size * 10
    f = [randint(1, 10) for _ in range(size)]
    n = [randint(1, 10) for _ in range(size)]
    matrix = [[0 for _ in range(size)] for j_i in range(size)]
    for i in range(size):
        matrix[i][i] = 941
        matrix[i][(i - 1) % size] = -n[i]
        matrix[i][(i + 1) % size] = -n[(i + 1) % size]

    a, t1 = cyclic_tridiagonal_solver(matrix, f)
    b, t2 = gauss_elimination(np.array(matrix), np.array(f))
    c, t3 = cholesky_solve(np.array(matrix), np.array(f))
    d, t4 = gradient_descent(np.array(matrix), np.array(f), np.zeros_like(f))
    print(max(b - a))
    print(max(c - a))
    print(max(d - a))
    print(f"Для матрицы размером {size} X {size}")
    print(f"{t1} - Метод циклической прогонки для,  {t2} - Ме-тод Гаусса,  {t3} - Метод Холецкого, {t4} - Метод наискорейшего спуска", sep = '\n')



