import math
import matplotlib.pyplot as plt  # 新增绘图库

# 定义方程和其导数
def f(x):
    return 9 * x - math.sin(x) - 1

def f_prime(x):
    return 9 - math.cos(x)

# 二分法
def bisection_method(a, b, iterations):
    results = []
    for i in range(iterations):
        c = (a + b) / 2
        results.append((i + 1, c, abs(b - a)))
        if f(c) == 0 or abs(b - a) < 1e-10:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return results

# 单点迭代法
def fixed_point_iteration(x0, iterations):
    results = []
    for i in range(iterations):
        x1 = -math.sin(x0) / 9 + 1 / 9
        results.append((i + 1, x1, abs(x1 - x0)))
        x0 = x1
    return results

# Newton 迭代法
def newton_method(x0, iterations):
    results = []
    for i in range(iterations):
        x1 = x0 - f(x0) / f_prime(x0)
        results.append((i + 1, x1, abs(x1 - x0)))
        x0 = x1
    return results

# 主函数
if __name__ == "__main__":
    iterations = 10

    # 初始化绘图
    plt.figure(figsize=(10, 6))
    plt.title('Error Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.yscale('log')

    # 二分法
    a, b = 0, 1
    bisection_results = bisection_method(a, b, iterations)
    iterations_bisection = [x[0] for x in bisection_results]
    errors_bisection = [x[2] for x in bisection_results]
    plt.plot(iterations_bisection, errors_bisection, 'o-', label='Bisection')

    # 单点迭代法
    x0 = 0.4
    fixed_point_results = fixed_point_iteration(x0, iterations)
    iterations_fixed = [x[0] for x in fixed_point_results]
    errors_fixed = [x[2] for x in fixed_point_results] 
    plt.plot(iterations_fixed, errors_fixed, 's--', label='Fixed-point')

    # Newton法
    x0 = 0.4
    newton_results = newton_method(x0, iterations)
    iterations_newton = [x[0] for x in newton_results]
    errors_newton = [x[2] for x in newton_results]
    plt.plot(iterations_newton, errors_newton, '^-', label='Newton')

    # 添加图例和保存
    plt.legend()
    plt.grid(True)
    plt.savefig('c:/Users/lgl20/Desktop/计算方法/convergence_plot.png')
    plt.close()
    for i, x, error in bisection_results:
        print(f"第{i}次迭代: x = {x:.10f}, 误差 = {error:.10f}")

    # 单点迭代法
    print("\n单点迭代法结果:")
    x0 = 0.4
    fixed_point_results = fixed_point_iteration(x0, iterations)
    for i, x, error in fixed_point_results:
        print(f"第{i}次迭代: x = {x:.10f}, 误差 = {error:.10f}")

    # Newton 迭代法
    print("\nNewton 迭代法结果:")
    x0 = 0.4
    newton_results = newton_method(x0, iterations)
    for i, x, error in newton_results:
        print(f"第{i}次迭代: x = {x:.10f}, 误差 = {error:.10f}")