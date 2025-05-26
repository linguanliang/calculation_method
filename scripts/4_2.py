import numpy as np
import matplotlib.pyplot as plt

# ================================================
# 问题 (1) 的微分方程和解析解
# ================================================
def problem1_rhs(x, y):
    """ dy/dx = (2/x)y + x^2 e^x """
    return (2/x)*y + x**2 * np.exp(x)

def problem1_exact(x):
    """ 解析解: y = x^2 (e^x - e) """
    return x**2 * (np.exp(x) - np.exp(1))

# ================================================
# 问题 (2) 的微分方程和解析解
# ================================================
def problem2_rhs(x, y):
    """ dy/dx = (1/x)(y^2 + y) """
    return (y**2 + y) / x

def problem2_exact(x):
    """ 解析解: y = 2x / (1 - 2x) """
    return 2*x / (1 - 2*x)

# ================================================
# 显式欧拉法数值求解
# ================================================
def euler_method(f, x0, y0, h, N):
    """ 显式欧拉法求解ODE """
    x = np.zeros(N+1)
    y = np.zeros(N+1)
    x[0], y[0] = x0, y0
    for i in range(N):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
    return x, y

# ================================================
# 参数设置与求解
# ================================================
x0, x_end = 1.0, 3.0  # 区间 [1, 3]
N_list = [5, 10, 20]  # 不同步长对应的分段数

# 存储结果
solutions = {
    'Problem1': {'exact': problem1_exact},
    'Problem2': {'exact': problem2_exact}
}

# 对每个问题和每个N进行求解
for N in N_list:
    h = (x_end - x0) / N  # 计算步长
    
    # 问题1求解
    x_num, y_num = euler_method(problem1_rhs, x0, 0.0, h, N)  # y(1)=0
    y_exact = problem1_exact(x_num)
    solutions['Problem1'][N] = (x_num, y_num, y_exact)
    
    # 问题2求解
    x_num, y_num = euler_method(problem2_rhs, x0, -2.0, h, N)  # y(1)=-2
    y_exact = problem2_exact(x_num)
    solutions['Problem2'][N] = (x_num, y_num, y_exact)

# ================================================
# 可视化与误差分析
# ================================================
def plot_results(problem, N_list, solutions):
    """ 绘制数值解、解析解及误差 """
    plt.figure(figsize=(12, 5))
    
    # 绘制数值解与解析解
    plt.subplot(1, 2, 1)
    x_fine = np.linspace(x0, x_end, 100)
    plt.plot(x_fine, solutions[problem]['exact'](x_fine), 'k-', label='Exact')
    for N in N_list:
        x_num, y_num, y_exact = solutions[problem][N]
        plt.plot(x_num, y_num, 'o--', label=f'N={N} (h={x_num[1]-x_num[0]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{problem}: Numerical vs Exact Solution')
    plt.legend()
    
    # 绘制绝对误差
    plt.subplot(1, 2, 2)
    for N in N_list:
        x_num, y_num, y_exact = solutions[problem][N]
        error = np.abs(y_num - y_exact)
        plt.semilogy(x_num, error, 'o--', label=f'N={N}')
    plt.xlabel('x')
    plt.ylabel('Absolute Error (log scale)')
    plt.title(f'{problem}: Error Analysis')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 显示结果
plot_results('Problem1', N_list, solutions)
plot_results('Problem2', N_list, solutions)