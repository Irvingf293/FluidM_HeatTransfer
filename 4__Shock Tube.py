import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, date

# %%
def time_IB():

    today_dt = date.today()
    time_i = datetime.now()
    time = time_i.strftime("%H:%M:%S")
    print("\nToday:", today_dt, '\nTime:', time, '\n')
    
    return time_i

def set_IC_shocktube(x_min, deltax, n, k):

    rho = 1.225
    p   = 101325 

    x = np.zeros((n,1))
    U = np.zeros((n,3))

    U_left       = np.zeros((1,3))
    U_left[0][0] = 4.0*rho
    U_left[0][2] = 4.0*p/(k-1)

    U_right       = np.zeros((1,3))
    U_right[0][0] = rho
    U_right[0][2] = p/(k-1)

    for i in range (1, n+1):
        x[i-1] = x_min + i*deltax
        
        if x[i-1] < 0.5:
            U[i-1] = U_left
        else:
            U[i-1] = U_right

    return  x, U, U_left, U_right

def find_delta_t(U, k, delta_x, CFL):

    n = len(U)
    delta_t = 1.0e42

    for i in range (0, n):
        rho = U[i][0]
        u   = U[i][1]/U[i][0]
        p   = (U[i][2] -0.5*U[i][1]**(2/U[i][0]))*(k-1)
        
        a = np.sqrt(k*p/rho)
        max_speed = np.abs(u) + a
        delta_t   = min(CFL*delta_x/max_speed, delta_t)
        
    return delta_t

def get_dUdt_Lax_Friedrichs(U, U_left, U_right, delta_x, delta_t, k):
        
    dUdt = np.zeros((len(U), len(U[0])))
    [F, F_left, F_right] = get_fluxes(U,U_left,U_right,k)

    n = len(U)
    dUdt[0]     = -(F[1]-F_left) / (2*delta_x) + 0.5*(U_left[0] - 2*U[0] +U[1])/delta_t
    
    for i in range (1, n-1):
        dUdt[i] = - (F[i+1] -  F[i-1])/ (2*delta_x) + 0.5*(U[i-1] - 2*U[i]   +U[i+1] )/delta_t
    dUdt[n-1]   = - (F_right - F[n-1])/ (2*delta_x) + 0.5*(U[n-1] - 2*U[n-1] +U_right)/delta_t

    return dUdt

def get_fluxes(U,U_left,U_right,k):

    F  = np.zeros((len(U), len(U[0])))
    F_left  = np.zeros(len(U_left[0]))
    F_right = np.zeros(len(U_left[0]))

    n = len(U)
    # Fill Force Matrix - F
    for i in range (0,n):
        rho = U[i][0]
        u   = U[i][1]/U[i][0]
        p   = (U[i][2] -0.5*U[i][1]**(2/U[i][0]))*(k-1)
    
        F[i][0] = rho*u
        F[i][1] = rho*u**2 + p
        F[i][2] = u*(k*p/(k-1) + 0.5*rho*u**2)

    # Fill Left Forse
    rho =  U_left[0][0]
    u   =  U_left[0][1] / U_left[0][0]
    p   = (U_left[0][2] - 0.5*(U_left[0][1])**2/U_left[0][0])*(k-1)

    F_left[0] = rho*u
    F_left[1] = rho*u**2 + p
    F_left[2] = u*(k*p/(k-1) + 0.5*rho*u**2)


    # Fill right Forse
    rho =  U_right[0][0]
    u   =  U_right[0][1] / U_right[0][0]
    p   = (U_right[0][2] - 0.5*(U_right[0][1])**2/U_right[0][0])*(k-1)

    F_right[0] = rho*u
    F_right[1] = rho*u**2 + p
    F_right[2] = u*(k*p/(k-1) + 0.5*rho*u**2)

    return F, F_left, F_right

def main():

    # Time
    time_i = time_IB()

    # Constants
    current_tm = 0.0        # Time
    final_time = 5.0e-4     # Output Time

    CFL = 0.99              # CFL Number
    n   = 1000              # Number of nodes in space                 
    k   = 1.4               # Ratio of Specific heats

    # Compute Delta_x
    x_min   = 0.0
    x_max   = 1.0
    delta_x = (x_max - x_min)/(n+1)

    [x, U, U_left, U_right] = set_IC_shocktube(x_min, delta_x, n, k)

    # Time Marching
    while current_tm < final_time:

        # Finde Delta_t
        delta_t = find_delta_t(U, k, delta_x, CFL)
        
        if (current_tm + delta_t) > final_time:
            delta_t = final_time - current_tm

        # Lax-Friedrich
        dUdt = get_dUdt_Lax_Friedrichs(U, U_left, U_right, delta_x, delta_t, k)
        U = U + delta_t*dUdt
        
        # Advance Time
        current_tm = current_tm + delta_t
    
    # Plot Density
    plt.figure(1, figsize=(10, 4))
    plt.plot(x, U[:,0])
    plt.title("ShockTube - Density")
    plt.xlabel("X")
    plt.ylabel("Density (Kg/m^3)")

    # Plot Velocity
    plt.figure(2, figsize=(10, 4))
    vel = U[:,1]/U[:,0]
    plt.plot(x, vel)
    plt.title("ShockTube - Velocity")
    plt.xlabel("X")
    plt.ylabel("Velocity (m/s)")

    # Plot Pressure
    plt.figure(3, figsize=(10, 4))
    pressure = (U[:,2] -0.5*U[:,1]*U[:,1]/U[:,0])*(k-1)
    plt.plot(x, pressure)
    plt.title("ShockTube - Pressure")
    plt.xlabel("X")
    plt.ylabel("Pa")
        

    time_f = datetime.now()
    time_d = time_f-time_i 
    print("\nRun Time: {}".format(time_f -time_i))

if __name__ == "__main__":
    main()
# %%