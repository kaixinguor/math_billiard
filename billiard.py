import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import argparse

# Superellipse equations and their gradients
def superellipse_K(x, y):
    """ Superellipse equation K: x^4 + y^4 = 1 """
    return x**4 + y**4 - 1

def superellipse_T(x, y):
    """ Superellipse equation T: x^(4/3) + y^(4/3) = 1 """
    return (x**4)**(1/3) + (y**4)**(1/3) - 1

def safe_cubic_root(x):
    if np.isnan(x) or np.isinf(x):
        return 0 # Or return some default value

    return np.cbrt(x)

def gradient_superellipse_T(x, y):
    """ Gradient of the superellipse T """
    df_dx = (4 / 3) * safe_cubic_root(x)
    df_dy = (4 / 3) * safe_cubic_root(y)

    return np.array([df_dx, df_dy])

def gradient_superellipse_K(x, y):
    """ Gradient of the superellipse K """
    df_dx = 4 * x**3
    df_dy = 4 * y**3
    return np.array([df_dx, df_dy])

# Finding intersection with superellipse using numerical methods (fsolve)
def find_intersection_with_superellipse(f, origin, direction):
    """ Find intersection of ray with superellipse by solving f(x, y) = 0 """
    def equations(vars):
        x, y = vars
        g = direction[0] * (y - origin[1]) - direction[1] * (x - origin[0])
        
        #print(f"iter:  f(x,y) = {f(x,y)}, g(x,y) = {g}")
        return [f(x, y), g]
    
    print(f"origin: {origin}, direction: {direction}")
    
    initial_guess = np.array([(origin[0] + direction[0])/2, (origin[1] + direction[1])/2], dtype=np.float64)
    print(f"initial guess: {initial_guess}")
    solution = fsolve(equations, initial_guess, xtol=1e-12, maxfev=5000)

    return solution

# Compute normal vectors to the superellipse at a given point
def compute_normal_to_superellipse(f, grad_f, point):
    """ Compute normal vector at point on the superellipse """
    grad = grad_f(point[0], point[1])
    return grad / np.linalg.norm(grad)

# Reverse direction if the ray is going outward (based on dot product with the normal)
def reverse_direction_if_outward(direction, normal):
    if np.dot(direction, normal) > 0:  # Direction is outward, reverse it
        return -direction
    return direction

# 模拟射线反弹过程
def simulate_bounce(p1, grad_S1, p2, grad_S2, iterations=1):
    """
    Simulate the bouncing process between two circles.
    """
    points_S1, points_S2 = [p1], [p2]
    normal_S1 = compute_normal_to_superellipse(None,grad_S1, p1)
    normal_S2 = compute_normal_to_superellipse(None, grad_S2, p2) # 从 p2 计算法向量，作为初始方向
    print("start: normal_S1: ", normal_S1, ", normal_S2: ", normal_S2)
    

    for iter in range(1, iterations+1):
        print(f"iter {iter}")

        # 从p1射线反弹
        dir2 = reverse_direction_if_outward(normal_S2, normal_S1)
        print(f"p1 start with direction: {dir2}")
        n1 = find_intersection_with_superellipse(superellipse_K, points_S1[-1], dir2)
        print("intersection n1:", n1)
        if n1 is None:
            break  # 如果没有交点，退出循环
        normal_S1 = compute_normal_to_superellipse(superellipse_K, gradient_superellipse_K, n1)
        points_S1.append(n1)

        # 对 p2 计算反弹
        # Reversing direction if it is outward
        dir1 = reverse_direction_if_outward(normal_S1, normal_S2)
        print(f"p2 start with direction: {dir1}")
        n2 = find_intersection_with_superellipse(superellipse_T, points_S2[-1], dir1)
        print("intersection n2:", n2)
        if n2 is None:
            break  # 如果没有交点，退出循环
        normal_S2 = compute_normal_to_superellipse(superellipse_T, gradient_superellipse_T, n2)
        points_S2.append(n2)

    return points_S1, points_S2

def plot(points_S1, points_S2):
    # 可视化结果
    # Plotting Curve K: x^4 + y^4 = 1
    x_K = np.linspace(-1, 1, 400)
    y_K = np.linspace(-1, 1, 400)

    X_K, Y_K = np.meshgrid(x_K, y_K)
    Z_K = X_K**4 + Y_K**4 - 1

    # Plotting Curve T: x^(4/3) + y^(4/3) = 1
    X_T, Y_T = np.meshgrid(x_K, y_K)
    # Z_T = X_T**(4/3) + Y_T**(4/3) - 1
    Z_T = (X_T**4)**(1/3) + (Y_T**4)**(1/3) - 1

    # Plotting
    plt.figure(figsize=(8, 8))

    # Contour for Curve K (x^4 + y^4 = 1)
    contour_K = plt.contour(X_K, Y_K, Z_K, levels=[0], colors='r')
    # Contour for Curve T (x^(4/3) + y^(4/3) = 1)
    contour_T = plt.contour(X_T, Y_T, Z_T, levels=[0], colors='b')

    plt.scatter(*zip(*points_S1), color="red", label="Bounce on S1")
    plt.scatter(*zip(*points_S2), color="blue", label="Bounce on S2")

    # Connect points on S1 with lines
    for i in range(len(points_S1) - 1):
        x_vals = [points_S1[i][0], points_S1[i + 1][0]]
        y_vals = [points_S1[i][1], points_S1[i + 1][1]]
        plt.plot(x_vals, y_vals, color="red", linestyle="--")

    # Connect points on S2 with lines (if you want to connect these as well)
    for i in range(len(points_S2) - 1):
        x_vals = [points_S2[i][0], points_S2[i + 1][0]]
        y_vals = [points_S2[i][1], points_S2[i + 1][1]]
        plt.plot(x_vals, y_vals, color="blue", linestyle="--")

    # Adding iteration labels
    for i, point in enumerate(points_S1):
        plt.text(point[0], point[1], f"iter {i}", color="red", fontsize=9, ha="left", va="bottom")
    
    for i, point in enumerate(points_S2):
        plt.text(point[0], point[1], f"iter {i}", color="blue", fontsize=9, ha="left", va="bottom")

    plt.legend()
    plt.axis("equal")
    plt.show()

def main(iterations):
    # Initial points on the superellipses
    p1 = np.array([0.5,(1-1./16)**(1/4)], dtype=np.float64)
    p2 = np.array([0.5,(1-0.5**(4/3))**(3/4)], dtype=np.float64)

    # Points lists for storing the trajectory
    points_S1, points_S2 = [p1], [p2]
    #plot(points_S1, points_S2)
    
    # Simulate and visualize
    points_S1, points_S2 = simulate_bounce(p1, gradient_superellipse_K, p2, gradient_superellipse_T, iterations=iterations)
    plot(points_S1, points_S2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Simulation with user-defined iterations.")
    parser.add_argument('--iter', type=int, default=10, help="Number of iterations (default: 10)")
    args = parser.parse_args()
    main(args.iter)
