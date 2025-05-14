import numpy as np
# 设置 numpy 输出格式，保留 4 位小数，不省略末尾 0
np.set_printoptions(formatter={'float': lambda x: f"{x:.4f}"})
def lu(A, b):
    n = len(A)
    L = np.eye(n)
    U = np.array(A, dtype=float)
    p = np.arange(n)
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[max_row, k]):
                max_row = i
        if max_row != k:
            U[[k, max_row], :] = U[[max_row, k], :]
            b[[k, max_row]] = b[[max_row, k]]
            p[[k, max_row]] = p[[max_row, k]]
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k + 1:] = U[i, k + 1:] - factor * U[k, k + 1:]
    return L, U, b, p
def forward_substitution(L, b):
    n = len(L)
    x = np.zeros(n)
    for k in range(n):
        x[k] = b[k]
        for j in range(k):
            x[k] -= L[k, j] * x[j]
        x[k] /= L[k, k]
    return x
def back_substitution(U, b):
    n = len(U)
    x = np.zeros(n)
    for k in range(n - 1, -1, -1):
        x[k] = b[k]
        for j in range(k + 1, n):
            x[k] -= U[k, j] * x[j]
        x[k] /= U[k, k]
    return x
def solve(A, b):
    L, U, b, p = lu(A, b)
    b = b[p]
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x


def Gauss_seidel(A, b, x0, tol=1e-4, max_iter=1000):
    n = len(A)
    x = x0.copy()
    print("\nGauss - Seidel Iteration Method:")
    print(f"Initial vector  x^0 = {x}")

    for iter_count in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        # 打印当前迭代结果（保留 4 位小数）
        print(f"Iteration {iter_count + 1}: x^{iter_count + 1} = {np.round(x_new, 4)}")

        # 判断是否收敛（使用四舍五入后的数值进行判断）
        if np.max(np.abs(x_new - x)) < tol:
            print(f"Gauss - Seidel Iteration Method reached error {tol} in {iter_count + 1} steps.")
            return x_new, iter_count + 1

        x = x_new

    print("Reached maximum iterations without full convergence.")
    return x, max_iter


def SOR(A, b, x0, w, max_iter=9):  # 修改为最多迭代9次
    n = len(A)
    x = x0.copy()
    print("\nSOR Iteration Method (Relaxation Factor ω = 1.2):")
    print(f"Initial vector  x^0 = {x}")

    print("\nSOR Iteration Format:")
    for i in range(n):
        expr = f"x^{{k+1}}_{{{i + 1}}} = (1-{w})x^{{k}}_{{{i + 1}}} + {w}×("
        for j in range(n):
            if j != i:
                sign = "+" if A[i, j] >= 0 else "-"
                coeff = abs(A[i, j])
                term = f"{coeff:.4f}x^{{{'k+1' if j < i else 'k'}}}_{{{j + 1}}}"
                expr += f"{sign} {term} "
        expr += f")/{A[i, i]:.4f}"
        print(expr)

    for iter_count in range(max_iter):  # 最多迭代9次
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - w) * x[i] + w * ((b[i] - s1 - s2) / A[i, i])
        print(f"Iteration {iter_count + 1}: x^{iter_count + 1} = {x_new}")
        x = x_new

    print(f"SOR Iteration Method (Relaxation Factor {w}) completed {max_iter} iterations.")
    return x, max_iter
# Define matrix A and vector b
A = np.array([
    [4, 1, -1],
    [2, 5, 1],
    [1, -2, 6]
])
b = np.array([3, 9, -4], dtype=float)
x0 = np.array([0.0, 0.0, 0.0])
# Solve using LU decomposition and output result
x_lu = solve(A, b)
print('\nSolution from Partial Pivoting Gaussian Elimination:', x_lu)
# Solve using Gauss-Seidel with convergence check
x_gauss, gauss_iter = Gauss_seidel(A, b, x0)
# Solve using SOR with fixed 9 iterations
x_sor, sor_iter = SOR(A, b, x0, w=1.2)
# Output final results
print('\nThe process has ended, exit code is 0')
