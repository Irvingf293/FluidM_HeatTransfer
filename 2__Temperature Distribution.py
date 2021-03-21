"""
    Author: Irving Barroso
    Tittle: Temperature Distribution
"""

# %%
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def source(x, y):
    if np.fabs(x) < 0.01 and np.fabs(y) < 0.01:
        return 125000.0
    else:
        return 0.0

def main():

    nx = 100
    ny = nx
    x_max = + 0.05
    x_min = - 0.05
    y_max = + 0.05
    y_min = - 0.05

    dx = (x_max - x_min)/(nx+1)
    dy = (y_max - y_min)/(ny+1)

    x_matrix = np.zeros((nx, ny))
    y_matrix = np.zeros((nx, ny))
    z_matrix = np.zeros((nx, ny))
    print('\nIB CFD - Temperature Distribution\n')

    for i in range(0, nx):
        for j in range(0, ny):
            x_matrix[i, j] = x_min + (i+1)*dx
            y_matrix[i, j] = y_min + (j+1)*dy

    #Boundary Temperature, Coefficient of thermal conductivity
    tb = 20.0
    k = 0.25
    num = nx * ny

    #Build A matrix, T, BC, and S, Allocate space:
    a = np.zeros((num, num))
    s = np.zeros((num, 1))
    bc = np.zeros((num, 1))

    print('Calculating... \n')
    for j in range(0, ny):
        for i in range(0, nx):

            # Computes global index and Diagonal term
            index = i + nx*(j-1)
            a[index, index] = k*(2.0/dx**2 + 2.0/dy**2)

            # x-direction terms
            if i == 0:
                bc[index] = bc[index] + k*tb/dx**2
            else:
                a[index, index-1] = - k/dx**2

            if i == nx:
                bc[index] = bc[index] + k*tb/dx**2
            else:
                a[index, index+1] = - k/dx**2

            #  y-direction terms
            if j == 0:
                bc[index] = bc[index] + k*tb/dy**2
            else:
                a[index, index-nx] = - k/dy**2

            if j == ny:
                bc[index] = bc[index] + k*tb/dy**2
            else:
                a[index, index+nx] = - k/dy**2


            # Find x and y coordinates and S
            x = x_matrix[i, j]
            y = y_matrix[i, j]
            s[index] = source(x, y)

            if ((x-0.025)**2+(y-0.025)**2)**0.5 <= 0.005:
                bc[index] = 5
                a[index, index-1] = 0
                a[index, index+1] = 0
                a[index, index-nx] = 0
                a[index, index+nx] = 0

            elif ((x+0.025)**2+(y+0.025)**2)**0.5 <= 0.005:
                bc[index] = 5
                a[index, index-1] = 0
                a[index, index+1] = 0
                a[index, index-nx] = 0
                a[index, index+nx] = 0

            if ((x+0.025)**2+(y-0.025)**2)**0.5 <= 0.005:
                bc[index] = 5
                a[index, index-1] = 0
                a[index, index+1] = 0
                a[index, index-nx] = 0
                a[index, index+nx] = 0

            elif ((x-0.025)**2+(y+0.025)**2)**0.5 <= 0.005:
                bc[index] = 5
                a[index, index-1] = 0
                a[index, index+1] = 0
                a[index, index-nx] = 0
                a[index, index+nx] = 0

    rhs = s + bc
    a_inv = np.linalg.inv(a)
    t = np.matmul(a_inv, rhs)

    print("nx = ", nx, "\ndx = ", "{:.6f}".format(dx), "\nT Max = ", "{:.2f}".format(np.max(t)))

    for j in range(0, ny):
        for i in range(0, nx):
            z_matrix[i, j] = t[i+(j-1)*nx]

    fig1 = plt.figure()
    countour = plt.contourf(x_matrix,y_matrix, z_matrix, cmap='coolwarm')
    fig1.suptitle("Temperature Distribution in 2D")
    fig1.colorbar(countour, shrink=1.0)

    fig2 = plt.figure()
    ax = fig2.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_matrix, y_matrix, z_matrix, cmap="coolwarm")
    fig2.suptitle("Temperature Distribution in 3D")
    fig2.colorbar(surf, shrink=0.5)
    plt.show()
    
if __name__ == '__main__':
    main()
# %%