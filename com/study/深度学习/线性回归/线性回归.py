import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(1, 20, 0.005)
y = np.arange(1, 20, 0.005)
X, Y = np.meshgrid(x, y)
Z = 1 + 2 * X - 3 * Y
ax.plot_surface(X, Y, Z, color='red')
ax.contourf(X, Y, Z, zdir='z', offset=0, alpha=0.4)
plt.show()