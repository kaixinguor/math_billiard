import numpy as np
import matplotlib.pyplot as plt

# 计算射线与圆的交点
def find_intersection_with_circle(center, radius, origin, direction):
    """
    Find intersection of a ray with a circle.
    center: Circle center (x, y)
    radius: Circle radius
    origin: Ray origin (x, y)
    direction: Ray direction (dx, dy)
    """
    cx, cy = center
    ox, oy = origin
    dx, dy = direction
    
    # Coefficients of the quadratic equation (t^2 * (dx^2 + dy^2) + 2t * ((ox - cx) * dx + (oy - cy) * dy) + ((ox - cx)^2 + (oy - cy)^2 - r^2) = 0)
    a = dx**2 + dy**2
    b = 2 * ((ox - cx) * dx + (oy - cy) * dy)
    c = (ox - cx)**2 + (oy - cy)**2 - radius**2
    
    # Solve the quadratic equation
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return None  # No intersection
    
    # Two solutions
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    
    # Choose the smallest positive t (intersection point in the forward direction)
    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    
    if t is None:
        return None  # No valid intersection
    
    # Return the intersection point
    intersection = origin + t * np.array([dx, dy])
    return intersection

# 计算法向量
def compute_normal_to_circle(center, point):
    """
    Compute the normal vector at a point on the circle.
    center: Circle center (x, y)
    point: Point on the circle (x, y)
    """
    normal = point - center
    return normal / np.linalg.norm(normal)  # Normalize the normal vector

# 反转射线方向
def reverse_direction_if_outward(direction, normal):
    if np.dot(direction, normal) > 0:  # Direction is outward, reverse it
        return -direction
    return direction

# 模拟射线反弹过程
def simulate_bounce(S1_center, S1_radius, S2_center, S2_radius, p1, p2, iterations=1):
    """
    Simulate the bouncing process between two circles.
    """
    points_S1, points_S2 = [p1], [p2]
    normal_S1 = compute_normal_to_circle(S1_center, p1)
    normal_S2 = compute_normal_to_circle(S2_center, p2) # 从 p2 计算法向量，作为初始方向
    print("start: normal_S1: ", normal_S1, ", normal_S2: ", normal_S2)
    

    for iter in range(iterations):
        print(f"iter {iter}")

        # 从p1射线反弹
        dir2 = reverse_direction_if_outward(normal_S2, normal_S1)
        print(f"p1 start with direction: {dir2}")
        n1 = find_intersection_with_circle(S1_center, S1_radius, points_S1[-1], dir2)
        print("intersection n1:", n1)
        if n1 is None:
            break  # 如果没有交点，退出循环
        normal_S1 = compute_normal_to_circle(S1_center, n1)
        points_S1.append(n1)

        # 对 p2 计算反弹
        # Reversing direction if it is outward
        dir1 = reverse_direction_if_outward(normal_S1, normal_S2)
        print(f"p2 start with direction: {dir1}")
        n2 = find_intersection_with_circle(S2_center, S2_radius, points_S2[-1], dir1)
        print("intersection n2:", n2)
        if n2 is None:
            break  # 如果没有交点，退出循环
        normal_S2 = compute_normal_to_circle(S2_center, n2)
        points_S2.append(n2)

    return points_S1, points_S2

def plot():
    # 可视化结果
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_S1 = S1_radius * np.array([np.cos(theta), np.sin(theta)]).T + S1_center
    circle_S2 = S2_radius * np.array([np.cos(theta), np.sin(theta)]).T + S2_center

    plt.plot(circle_S1[:, 0], circle_S1[:, 1], label="S1")
    plt.plot(circle_S2[:, 0], circle_S2[:, 1], label="S2")
    plt.scatter(*zip(*points_S1), color="red", label="Bounce on S1")
    plt.scatter(*zip(*points_S2), color="blue", label="Bounce on S2")
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    # 圆心和半径
    S1_center = np.array([-2, 0])
    S1_radius = 1
    S2_center = np.array([2, 0])
    S2_radius = 1

    # 初始点和方向
    p1 = np.array([-2, 1])  # S1上的起点
    p2 = np.array([2+1/np.sqrt(2), 1/np.sqrt(2)])  # S2上的起点

    points_S1, points_S2 = [p1], [p2]
    plot()

    # 进行模拟
    points_S1, points_S2 = simulate_bounce(S1_center, S1_radius, S2_center, S2_radius, p1, p2, iterations=2)

    plot()