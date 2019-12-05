import numpy as np
import math
from math import exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def s4(b, c, t):
    return exp(-b * t) + exp(-c * t)

B = 3
C = 5
array_b = np.arange(B - 1, B + 1, 0.0625)
nb = len(array_b)
array_c = np.arange(C - 2, C + 2, 0.125)
nc = len(array_c)
array_t = np.arange(1, 10, 1)
E = np.zeros([nb, nc])
for i in range(nb):
    for j in range(nc):
        E_ti = 0
        bi = array_b[i]
        ci = array_c[j]
        for ti in array_t:
            fbc = s4(bi, ci, ti)
            ft = s4(B, C, ti)
            E_ti += pow(ft - fbc, 2)
        E[i, j] = E_ti

# Emin 
Emin = np.amin(E)
pos_min = np.where(E == np.amin(E))
i_min = pos_min[0][0]
j_min = pos_min[1][0]

# Emax 
Emax = np.amax(E)
pos_max = np.where(E == np.amax(E))
i_max = pos_max[0][0]
j_max = pos_max[1][0]

# Plot a 3D surface
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(array_b, array_c, E, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('b')
ax.set_ylabel('c')
ax.set_zlabel('E')
ax.set_xlim(B - 1.2, B + 1.2)
ax.set_ylim(C - 2.2, C + 2.2)
ax.set_zlim(Emin, Emax)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# dots in surface
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(array_b, array_c, E, cmap='Purples_r')
ax.set_xlabel('b')
ax.set_ylabel('c')
ax.set_zlabel('E')
ax.set_zlim(Emin, Emax)
ax.scatter3D(array_b[i_min], array_c[j_min], Emin, c='red', marker='o')
ax.scatter3D(array_b[i_max], array_c[j_max], Emax, c='black', marker='o')
# Edots
for i in range(1, i_min):
    ax.scatter3D(array_b[i], array_c[i], E[i, i], s=20, c='black', marker='o')
plt.title('Surface E')

# Plot a 3D scontour

fig = plt.figure()
ax = fig.gca()
ax.contour(array_b, array_c, E, 200, origin='lower', cmap='Purples_r', linewidths=2, extent=(-3, 3, -2, 1))
plt.plot(array_b[i_min], array_c[j_min], c='red', marker='o')
plt.plot(array_b[i_max], array_c[i_max], c='black', marker='o')
for i in range(1, i_min):
    if (math.fmod(i, 4) == 0):
        plt.plot(array_b[i], array_c[i], c='black', marker='o')
ax.set_xlim(B - 1.2, B + 1.2)
ax.set_ylim(C - 2.2, C + 2.2)
plt.title('Contour plot of surface E')

# Print E (printE) with array_b in [2.7, 3.3], array_c in [4.4, 5.6]
printE = np.zeros([10, 10])
printE = E[10:23, 10:23]
plt.figure()
plt.plot(printE)
plt.title('printE')
