import numpy as np

def chebyshev_nodes(n):
    """生成切比雪夫插值节点"""
    return np.cos([(2*i + 1)*np.pi / (2*(n + 1)) for i in range(n+1)])

def lagrange_interp(x, nodes, f_vals):
    """计算拉格朗日插值多项式在x处的值"""
    n = len(nodes)
    result = 0.0
    for i in range(n):
        term = f_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - nodes[j]) / (nodes[i] - nodes[j])
        result += term
    return result

# 需要计算的x点列表
target_x = [-0.95, -0.05, 0.05, 0.95]

# 问题1: f(x) = 1/(1+x²)
print("问题1: f(x) = 1/(1+x²)")
for n in [5, 10, 20]:
    # 生成节点和函数值
    nodes = chebyshev_nodes(n)
    f_values = 1 / (1 + nodes**2)
    
    print(f"\nn = {n}:")
    for x in target_x:
        # 计算插值
        Ln_x = lagrange_interp(x, nodes, f_values)
        true_value = 1 / (1 + x**2)
        error = abs(Ln_x - true_value)
        print(f"x={x:.2f}: Ln(x)={Ln_x:.6f}, 误差={error:.2e}")

# 问题2: f(x) = e^x
print("\n\n问题2: f(x) = e^x")
for n in [5, 10, 20]:
    # 生成节点和函数值
    nodes = chebyshev_nodes(n)
    f_values = np.exp(nodes)
    
    print(f"\nn = {n}:")
    for x in target_x:
        # 计算插值
        Ln_x = lagrange_interp(x, nodes, f_values)
        true_value = np.exp(x)
        error = abs(Ln_x - true_value)
        print(f"x={x:.2f}: Ln(x)={Ln_x:.6f}, 误差={error:.2e}")