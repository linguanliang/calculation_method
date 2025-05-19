import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def lagrange_interpolation(x_nodes, y_nodes, x):
    """手动实现拉格朗日插值公式"""
    n = len(x_nodes)
    result = 0.0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

def run_unified_experiment():
    """统一实验框架，同时处理两个问题"""
    # 实验配置
    experiments = {
        "问题1: Runge函数": {
            "f": lambda x: 1/(1 + x**2),
            "interval": (-5, 5),
            "test_points": [0.75, 1.75, 2.75, 3.75, 4.75],
            "ns": [5, 10, 20]
        },
        "问题2: 指数函数": {
            "f": np.exp,
            "interval": (-5, 5),
            "test_points": [-4.75, -0.25, 0.25, 4.75],
            "ns": [5, 10, 20]
        }
    }

    # 遍历两个实验
    for exp_name, config in experiments.items():
        a, b = config["interval"]
        f = config["f"]
        ns = config["ns"]
        test_points = config["test_points"]
        
        plt.figure(figsize=(12, 6))
        print(f"\n{'='*40}\n{exp_name}实验结果\n{'='*40}")

        # 遍历每个n值
        for n in ns:
            # 使用切比雪夫节点
            x_nodes = [0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * (n + 1))) for i in range(n + 1)]
            y_nodes = [f(x) for x in x_nodes]
            
            # 计算测试点插值
            interpolated = [lagrange_interpolation(x_nodes, y_nodes, x) for x in test_points]
            true_values = [f(x) for x in test_points]
            errors = [abs(ip - tv) for ip, tv in zip(interpolated, true_values)]
            
            # 生成密集点绘图
            x_dense = np.linspace(a, b, 500)
            y_interp = [lagrange_interpolation(x_nodes, y_nodes, x) for x in x_dense]
            
            # 绘制插值曲线
            plt.plot(x_dense, y_interp, '--', alpha=0.8, label=f'n={n}')
            
            # 打印数值结果
            print(f"\nn={n}:")
            for x, ip, err in zip(test_points, interpolated, errors):
                print(f"x={x:6.2f} | 插值值={ip:12.6f} | 误差={err:12.4e}")

        # 绘制真实函数和测试点
        y_true = [f(x) for x in x_dense]
        plt.plot(x_dense, y_true, 'k-', label='真实函数')
        plt.scatter(test_points, [f(x) for x in test_points], 
                   color='red', zorder=5, label='测试点')
        plt.title(f"{exp_name} 插值对比 (切比雪夫节点)")
        plt.legend()
        plt.show()

# 执行统一实验
if __name__ == "__main__":
    run_unified_experiment()