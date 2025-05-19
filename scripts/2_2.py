import numpy as np

def create_matrix(n, a):
    """创建n维矩阵，对角线元素为2，非对角线元素为a"""
    mat = np.full((n, n), a)
    np.fill_diagonal(mat, 2)
    return mat

def jacobi(A, b, x0=None, max_iter=1000, tol=1e-6):
    """雅可比迭代法实现"""
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    D = np.diag(A)
    LU = A - np.diagflat(D)
    errors = []
    
    for k in range(max_iter):
        x_new = (b - np.dot(LU, x)) / D
        error = np.linalg.norm(np.dot(A, x_new) - b)
        errors.append(error)
        if error < tol:
            return x_new, errors, True, k+1
        x = x_new
    return x, errors, False, max_iter

def gauss_seidel(A, b, x0=None, max_iter=1000, tol=1e-6):
    """高斯-赛德尔迭代法实现"""
    n = len(b)
    x = x0.copy() if x0 is not None else np.zeros(n)
    errors = []
    
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x_new[i+1:])) / A[i, i]
        error = np.linalg.norm(np.dot(A, x_new) - b)
        errors.append(error)
        if error < tol:
            return x_new, errors, True, k+1
        x = x_new
    return x, errors, False, max_iter

# 参数设置
n = 3                   # 矩阵维度
true_x = np.ones(n)     # 假设真实解为全1向量
x0 = np.zeros(n)        # 初始猜测值
max_iter = 1000         # 最大迭代次数
tol = 1e-6              # 收敛阈值

# 实验1：雅可比方法收敛
a_j_conv = 0.5
A = create_matrix(n, a_j_conv)
b = A @ true_x  # 生成对应真实解的b向量
x_j, errors_j, conv_j, steps_j = jacobi(A, b, x0, max_iter, tol)
print(f"当a={a_j_conv}时，雅可比方法收敛：{conv_j}")
print(f"迭代步数：{steps_j}，最终误差：{errors_j[-1]:.2e}\n")

# 实验2：雅可比方法发散
a_j_div = 1.5
A = create_matrix(n, a_j_div)
b = A @ true_x
_, errors_j_div, _, _ = jacobi(A, b, x0, max_iter=5, tol=tol)
print(f"当a={a_j_div}时，雅可比方法前5次误差：{[f'{e:.2e}' for e in errors_j_div[:5]]}\n")

# 实验3：高斯-赛德尔方法收敛
a_gs_conv = 0.5
A = create_matrix(n, a_gs_conv)
b = A @ true_x
x_gs, errors_gs, conv_gs, steps_gs = gauss_seidel(A, b, x0, max_iter, tol)
print(f"当a={a_gs_conv}时，高斯-赛德尔收敛：{conv_gs}")
print(f"迭代步数：{steps_gs}，最终误差：{errors_gs[-1]:.2e}\n")

# 实验4：高斯-赛德尔方法发散
a_gs_div = 1.5
A = create_matrix(n, a_gs_div)
b = A @ true_x
_, errors_gs_div, _, _ = gauss_seidel(A, b, x0, max_iter=5, tol=tol)
print(f"当a={a_gs_div}时，高斯-赛德尔方法前5次误差：{[f'{e:.2e}' for e in errors_gs_div[:5]]}\n")

# 实验5：两种方法均收敛
a_both = 0.5
A = create_matrix(n, a_both)
b = A @ true_x
_, _, _, steps_j = jacobi(A, b, x0, max_iter, tol)
_, _, _, steps_gs = gauss_seidel(A, b, x0, max_iter, tol)
print(f"当a={a_both}时，雅可比收敛步数：{steps_j}，高斯-赛德尔收敛步数：{steps_gs}")
print("高斯-赛德尔收敛更快" if steps_gs < steps_j else "雅可比收敛更快")