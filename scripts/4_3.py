import numpy as np
import matplotlib.pyplot as plt

# ================================================
# 定义三个问题的微分方程和解析解
# ================================================
def problem1_rhs(x, y):
    return -20*(y - x**2) + 2*x

def problem1_exact(x):
    return x**2 + (1/3)*np.exp(-20*x)

def problem2_rhs(x, y):
    return -20*y + 20*np.sin(x) + np.cos(x)

def problem2_exact(x):
    return np.exp(-20*x) + np.sin(x)

def problem3_rhs(x, y):
    return -20*(y - np.exp(x)*np.sin(x)) + np.exp(x)*(np.sin(x) + np.cos(x))

def problem3_exact(x):
    return np.exp(x)*np.sin(x)

# ================================================
# 显式欧拉法求解
# ================================================
def euler_method(f, x0, y0, h, N):
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0], y[0] = x0, y0
    for i in range(N):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
    return x, y

# ================================================
# 参数设置与计算
# ================================================
x0, x_end = 0.0, 1.0
N_list = [5, 10, 20]  # 对应 h=0.2, 0.1, 0.05

# 存储结果
problems = {
    'Problem1': {'rhs': problem1_rhs, 'exact': problem1_exact, 'y0': 1/3},
    'Problem2': {'rhs': problem2_rhs, 'exact': problem2_exact, 'y0': 1},
    'Problem3': {'rhs': problem3_rhs, 'exact': problem3_exact, 'y0': 0}
}

solutions = {}
for pname in problems:
    solutions[pname] = {}
    p = problems[pname]
    for N in N_list:
        h = (x_end - x0) / N
        x_num, y_num = euler_method(p['rhs'], x0, p['y0'], h, N)
        y_exact = p['exact'](x_num)
        solutions[pname][N] = (x_num, y_num, y_exact)

# ================================================
# 可视化
# ================================================
def plot_problem(pname, solutions):
    plt.figure(figsize=(12, 5))
    
    # 绘制数值解与解析解
    plt.subplot(1, 2, 1)
    x_fine = np.linspace(x0, x_end, 100)
    plt.plot(x_fine, problems[pname]['exact'](x_fine), 'k-', label='Exact')
    for N in N_list:
        x_num, y_num, _ = solutions[pname][N]
        plt.plot(x_num, y_num, 'o--', markersize=4, label=f'N={N} (h={x_num[1]-x_num[0]:.2f})')
    plt.xlabel('x'), plt.ylabel('y')
    plt.title(f'{pname}: Numerical vs Exact Solution')
    plt.legend()
    
    # 绘制误差（对数坐标）
    plt.subplot(1, 2, 2)
    for N in N_list:
        x_num, y_num, y_exact = solutions[pname][N]
        error = np.abs(y_num - y_exact)
        plt.semilogy(x_num, error, 'o--', label=f'N={N}')
    plt.xlabel('x'), plt.ylabel('Absolute Error (log scale)')
    plt.title(f'{pname}: Error Analysis')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 绘制所有问题的结果
for pname in problems:
    plot_problem(pname, solutions)