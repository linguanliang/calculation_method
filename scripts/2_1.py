import numpy as np
import sympy as sp
from scipy.linalg import eigvals

# 符号计算部分（使用SymPy）
def symbolic_analysis():
    # 定义符号变量
    a = sp.symbols('a')
    
    # 定义矩阵 A
    A = sp.Matrix([
        [1, a, a],
        [a, 1, a],
        [a, a, 1]
    ])
    
    # 提取下三角矩阵并取负
    L = sp.zeros(*A.shape)  # 创建一个与 A 形状相同的零矩阵
    for i in range(A.rows):
        for j in range(A.cols):
            if i >= j:  # 仅保留下三角部分
                L[i, j] = -A[i, j]
    
    print("下三角矩阵 L:")
    sp.pprint(L)

# 数值验证部分（使用NumPy）
def numerical_validation(a_value):
    A = np.array([[1, a_value, a_value],
                 [a_value, 1, a_value],
                 [a_value, a_value, 1]], dtype=float)
    
    # 验证正定性
    is_positive_definite = np.all(np.linalg.eigvals(A) > 0)
    print(f"\n当a={a_value}时，矩阵正定: {is_positive_definite}")
    
    # 构造雅可比迭代矩阵
    D = np.diag(np.diag(A))
    L_plus_U = D - A
    Bj = np.linalg.inv(D) @ L_plus_U
    rho_j = np.max(np.abs(eigvals(Bj)))
    print(f"雅可比迭代谱半径: {rho_j:.4f} (收敛: {rho_j < 1})")

    # 构造高斯-赛德尔迭代矩阵
    L = np.tril(A, -1)
    U = A - np.tril(A)
    Bgs = np.linalg.inv(D - L) @ U
    rho_gs = np.max(np.abs(eigvals(Bgs)))
    print(f"高斯-赛德尔谱半径: {rho_gs:.4f} (收敛: {rho_gs < 1})")

if __name__ == "__main__":
    print("=== 符号计算分析 ===")
    symbolic_analysis()
    
    print("\n=== 数值验证示例 ===")
    numerical_validation(0.5)  # 收敛情况
    numerical_validation(2.0)  # 发散情况