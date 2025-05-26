import numpy as np
import matplotlib.pyplot as plt

def solve_problem1(N):
    a, b = 0, 1
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y_euler = np.zeros(N+1)
    y_euler[0] = -1  # 初始条件
    
    for i in range(N):
        y_euler[i+1] = y_euler[i] + h * (x[i] + y_euler[i])
    
    y_analytic = -x - 1
    return x, y_euler, y_analytic

# 绘制结果
Ns = [5, 10, 20]
plt.figure(figsize=(10, 6))
for N in Ns:
    x, y_euler, y_analytic = solve_problem1(N)
    plt.plot(x, y_euler, 'o--', label=f'N={N} 数值解')
    plt.plot(x, y_analytic, 'k-', lw=1, alpha=0.5)

plt.title("问题(1): 数值解与解析解对比（完全重合）")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
def solve_problem2(N):
    a, b = 0, 1
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y_euler = np.zeros(N+1)
    y_euler[0] = 1  # 初始条件
    
    for i in range(N):
        y_euler[i+1] = y_euler[i] + h * (-y_euler[i]**2)
    
    y_analytic = 1 / (x + 1)
    return x, y_euler, y_analytic

# 绘制结果
Ns = [5, 10, 20]
plt.figure(figsize=(10, 6))
for N in Ns:
    x, y_euler, y_analytic = solve_problem2(N)
    plt.plot(x, y_euler, 'o--', label=f'N={N} 数值解')
    plt.plot(x, y_analytic, 'k-', lw=1, alpha=0.5)

plt.title("问题(2): 数值解与解析解对比（存在误差）")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()