import math
import time
import matplotlib.pyplot as plt

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

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 初始化绘图
    plt.figure(figsize=(10, 6))
    plt.title('牛顿法误差收敛对比')
    plt.xlabel('迭代次数')
    plt.ylabel('误差（对数坐标）')
    plt.yscale('log')

    # 收集三种方法的误差数据
    methods = {
        "标准牛顿法": newton_method(x0, max_iter)[0],
        "已知重数修正法": modified_newton_known_m(x0, max_iter)[0],
        "未知重数修正法": modified_newton_unknown_m(x0, max_iter)[0]
    }

    # 绘制曲线
    markers = ['o-', 's--', '^-']
    for (name, results), marker in zip(methods.items(), markers):
        iterations = [x[0] for x in results]
        errors = [x[2] for x in results]
        plt.plot(iterations, errors, marker, label=name)

    # 添加图例和保存
    plt.legend()
    plt.grid(True)
    plt.savefig('c:/Users/lgl20/Desktop/计算方法/newton_comparison.png')
    plt.close()

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