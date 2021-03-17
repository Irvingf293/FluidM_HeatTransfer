"""
    Author: Irving Barroso
    Tittle: Intro to Finite Difference Approximation

"""

# %% 
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#   Book: Fundamental of Computational Fluid Dynamics >>> Chapter: 3.0
# ----------------------------------------------------------------------

# Given Function
def y(x):
    return (x**3)*(np.sin(x))

# First, Second and Third Exact Derivative
def dy_dx1(x):
    return (3*x**2)*(np.sin(x)) + (x**3)*(np.cos(x))

def dy_dx2(x):
    return (6*x**2)*(np.cos(x)) + (6*x - x**3)*(np.sin(x))

def dy_dx3(x):
    return (np.sin(x))*(6 - 9*x**2) + (np.cos(x))*(18*x - x**3)

# First, Second and Third Order Approximation
def dy1(x, dx):
    return (1/dx)*(-(11/6.0)*(y(x)) + 3*(y(x+dx)) + (-3/2.0)*(y(x+2*dx)) + (1/3.0)*(y(x+3*dx)))

def dy2(x, dx):
    return (1/(dx**2))*(+ 2*y(x) - 5*y(x+dx) + 4*y(x+2*dx) - 1*y(x+3*dx))

def dy3(x, dx):
    return (1/(dx**3))*(-1*y(x) + 3*(y(x+dx)) - 3*(y(x+2*dx)) + y(x+3*dx))

def main():

    # -------------------------------------------------
    #                         Part I
    # -------------------------------------------------
    print('\nIntro to Finite Difference Approximation')
    print('Function y = (x^3)*Sin(x)\n')
    x = 5
    dx1 = 0.5
    dx2 = dx1 / 2
    dx3 = dx2 / 2
    dx4 = dx3 / 2
    dx5 = dx4 / 2

    #   Exact 1st, 2nd and 3rd Derivative of y at x = 5
    exact1 = dy_dx1(x)
    exact2 = dy_dx2(x)
    exact3 = dy_dx3(x)

    print("y'(5)   =",  exact1)
    print("y''(5)  =", exact2)
    print("y'''(5) =",  exact3)

    #   1st Order Method Approximation
    approx11 = dy1(x, dx1)
    approx12 = dy1(x, dx2)
    approx13 = dy1(x, dx3)
    approx14 = dy1(x, dx4)
    approx15 = dy1(x, dx5)

    #   Errors Using First Order Method
    error11 = np.fabs(exact1 - approx11)
    error12 = np.fabs(exact1 - approx12)
    error13 = np.fabs(exact1 - approx13)
    error14 = np.fabs(exact1 - approx14)
    error15 = np.fabs(exact1 - approx15)

    #   Order of convergence
    order11 = np.log(error12 / error11) / np.log(dx2 / dx1)
    order12 = np.log(error13 / error12) / np.log(dx3 / dx2)
    order13 = np.log(error14 / error13) / np.log(dx4 / dx3)
    order14 = np.log(error15 / error14) / np.log(dx5 / dx4)

    print("\n\nFirst Order Finite Difference ")
    print("Errors \t\t\t\t Convergence")
    print(error11)
    print(error12, "\t\t", order11)
    print(error13, "\t\t", order12)
    print(error14, "\t\t", order13)
    print(error15, "\t\t", order14)

    # --------------------------------------------------
    #             2nd Order Method Approximation 
    # --------------------------------------------------
    approx21 = dy2(x, dx1)
    approx22 = dy2(x, dx2)
    approx23 = dy2(x, dx3)
    approx24 = dy2(x, dx4)
    approx25 = dy2(x, dx5)

    #   Errors Using Second - Order Method
    error21 = np.fabs(exact2 - approx21)
    error22 = np.fabs(exact2 - approx22)
    error23 = np.fabs(exact2 - approx23)
    error24 = np.fabs(exact2 - approx24)
    error25 = np.fabs(exact2 - approx25)

    #   Order of convergence
    order21 = np.log(error22 / error21) / np.log(dx2 / dx1)
    order22 = np.log(error23 / error22) / np.log(dx3 / dx2)
    order23 = np.log(error24 / error23) / np.log(dx4 / dx3)
    order24 = np.log(error25 / error24) / np.log(dx5 / dx4)

    print("\n\nSecond Order Finite Difference ")
    print("Errors \t\t\t\t Convergence")
    print(error21)
    print(error22, "\t\t", order21)
    print(error23, "\t\t", order22)
    print(error24, "\t\t", order23)
    print(error25, "\t\t", order24)

    # --------------------------------------------------
    #            3rd Order Method Approximation 
    # --------------------------------------------------
    approx31 = dy3(x, dx1)
    approx32 = dy3(x, dx2)
    approx33 = dy3(x, dx3)
    approx34 = dy3(x, dx4)
    approx35 = dy3(x, dx5)

    #   Errors Using Second - Order Method
    error31 = np.fabs(exact3 - approx31)
    error32 = np.fabs(exact3 - approx32)
    error33 = np.fabs(exact3 - approx33)
    error34 = np.fabs(exact3 - approx34)
    error35 = np.fabs(exact3 - approx35)

    #   Order of convergence
    order31 = np.log(error32 / error31) / np.log(dx2 / dx1)
    order32 = np.log(error33 / error32) / np.log(dx3 / dx2)
    order33 = np.log(error34 / error33) / np.log(dx4 / dx3)
    order34 = np.log(error35 / error34) / np.log(dx5 / dx4)

    print("\n\nThird Order Finite Difference ")
    print("Errors \t\t\t\t Convergence")
    print(error31)
    print(error32, "\t\t", order31)
    print(error33, "\t\t", order32)
    print(error34, "\t\t", order33)
    print(error35, "\t\t", order34, "\n\n")

    # ---------------------------------------------
    #                 Part II
    # ---------------------------------------------

    #   Select the number of points in the plot (n)
    #   Create matrix nx1 to store information
    n = 30

    dxs = np.zeros((n, 1))
    log_inv_dxs = np.zeros((n, 1))

    log_errors1 = np.zeros((n, 1))
    log_errors2 = np.zeros((n, 1))
    log_errors3 = np.zeros((n, 1))

    errors1 = np.zeros((n, 1))
    errors2 = np.zeros((n, 1))
    errors3 = np.zeros((n, 1))

    # Loop throughout to fill "dxs", "errors1", "errors2"
    # Each time through the loop, Delta x is half as big
    for i in range(0, n):
        dxs[i] = 0.5 ** i
        errors1[i] = np.fabs(exact1 - dy1(x, dxs[i]))
        errors2[i] = np.fabs(exact2 - dy2(x, dxs[i]))
        errors3[i] = np.fabs(exact3 - dy3(x, dxs[i]))

        # Compute the log of the inverse of delta x
        log_inv_dxs[i] = np.log10(1.0 / dxs[i])

        # Compute the log of the errors
        log_errors1[i] = np.log10(errors1[i])
        log_errors2[i] = np.log10(errors2[i])
        log_errors3[i] = np.log10(errors3[i])

    # Compute reference lines with the expected slope
    ref_line1 = -3 * log_inv_dxs - 2
    ref_line2 = -2 * log_inv_dxs - 2
    ref_line3 = -1 * log_inv_dxs - 2

    plt.subplot(3, 1, 1)
    plt.title("Log-log plot of the error vs grid size")
    plt.plot(log_inv_dxs, ref_line1, color="blue")
    plt.plot(log_inv_dxs, log_errors1, color="red", lw="2", ls="--")

    plt.subplot(3, 1, 2)
    plt.ylabel("log10(error")
    plt.plot(log_inv_dxs, ref_line2, color="blue")
    plt.plot(log_inv_dxs, log_errors2, color="red", lw="2", ls="--")

    plt.subplot(3, 1, 3)
    plt.xlabel("log10(1/Delta(x)")
    plt.plot(log_inv_dxs, ref_line3, color="blue")
    plt.plot(log_inv_dxs, log_errors3, color="red", lw="2", ls="--")
    plt.show()

if __name__ == '__main__':
    main()

# %%