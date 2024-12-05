import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Generate values for x and y
theta = np.linspace(0, 2 * np.pi, 1000)
x_vals_K = np.cos(theta)
y_vals_K = np.sin(theta)

x_vals_T = np.cos(theta)
y_vals_T = np.sin(theta)

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
contour_K = plt.contour(X_K, Y_K, Z_K, levels=[0], colors='b')
# Contour for Curve T (x^(4/3) + y^(4/3) = 1)
contour_T = plt.contour(X_T, Y_T, Z_T, levels=[0], colors='r')

# Add the legend manually
handles = [Line2D([0], [0], color='b', label='K: x^4 + y^4 = 1'),
           Line2D([0], [0], color='r', label='T: x^(4/3) + y^(4/3) = 1')]
plt.legend(handles=handles)

# Labeling
plt.title("Superellipse Comparison: K and T")
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Show plot
plt.show()
