import math
import time

# 定义目标函数及其导数
def f(x):
    return (math.sin(x) - x / 2) ** 2

def f_prime(x):
    return 2 * (math.sin(x) - x / 2) * (math.cos(x) - 1 / 2)

def f_double_prime(x):
    return 2 * ((math.cos(x) - 1 / 2) ** 2 + (math.sin(x) - x / 2) * (-math.sin(x)))

# Newton 迭代法
def newton_method(x0, max_iter=10):
    results = []
    start_time = time.time()
    x = x0
    for i in range(max_iter):
        x_new = x - f(x) / f_prime(x)
        error = abs(x_new - x)
        results.append((i + 1, x_new, error))
        x = x_new
    elapsed_time = time.time() - start_time
    return results, elapsed_time

# 已知重数的修正 Newton 迭代法 (重数 m = 2)
def modified_newton_known_m(x0, m=2, max_iter=10):
    results = []
    start_time = time.time()
    x = x0
    for i in range(max_iter):
        f_prime_val = f_prime(x)
        if abs(f_prime_val) < 1e-10:  # 检查导数是否接近零
            print(f"警告: 在迭代 {i + 1} 时，f_prime(x) 接近零，跳过此迭代。")
            break
        x_new = x - m * f(x) / f_prime_val
        error = abs(x_new - x)
        results.append((i + 1, x_new, error))
        x = x_new
    elapsed_time = time.time() - start_time
    return results, elapsed_time

# 未知重数的修正 Newton 迭代法
def modified_newton_unknown_m(x0, max_iter=10):
    results = []
    start_time = time.time()
    x = x0
    for i in range(max_iter):
        m = f(x) * f_double_prime(x) / (f_prime(x) ** 2)  # 动态计算重数
        x_new = x - f(x) / (m * f_prime(x))
        error = abs(x_new - x)
        results.append((i + 1, x_new, error))
        x = x_new
    elapsed_time = time.time() - start_time
    return results, elapsed_time

# 主程序
if __name__ == "__main__":
    x0 = math.pi / 2  # 初始值
    max_iter = 10

    # Newton 迭代法
    results_newton, time_newton = newton_method(x0, max_iter)
    print("Newton 迭代法:")
    for i, x, error in results_newton:
        print(f"迭代 {i}: x = {x:.10f}, 误差 = {error:.10e}")
    print(f"计算时间: {time_newton:.10f} 秒\n")

    # 已知重数的修正 Newton 迭代法
    results_modified_known, time_modified_known = modified_newton_known_m(x0, m=2, max_iter=max_iter)
    print("已知重数的修正 Newton 迭代法:")
    for i, x, error in results_modified_known:
        print(f"迭代 {i}: x = {x:.10f}, 误差 = {error:.10e}")
    print(f"计算时间: {time_modified_known:.10f} 秒\n")

    # 未知重数的修正 Newton 迭代法
    results_modified_unknown, time_modified_unknown = modified_newton_unknown_m(x0, max_iter)
    print("未知重数的修正 Newton 迭代法:")
    for i, x, error in results_modified_unknown:
        print(f"迭代 {i}: x = {x:.10f}, 误差 = {error:.10e}")
    print(f"计算时间: {time_modified_unknown:.10f} 秒")