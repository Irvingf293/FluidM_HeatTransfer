import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

# %%

def exact_time_hist(t_fin, n):
    dt = t_fin/(n-1)
    exact_times = np.zeros((n, 1))
    exact_sol   = np.zeros((n, 1))

    for i in range(0, n):
        exact_times[i] = dt*(i)
        exact_sol[i] = exact(exact_times[[i]])

    return exact_times, exact_sol

def exact(t):
    return np.cos(3*t)*(1+(5*np.exp(-t/2)))
    
def forcing(t):
    return -3*np.sin(3*t)*(1+5*np.exp(-t/2)) +0.5*np.cos(3*t)

def main():

 
    print('IB CFD - Time Marching Method ')


    # Output time, time step and exact number
    n = 70
    t_final = 4.0
    exact_num = 5000

    # Get exact soluton for referece
    [exact_times, exact_sol] = exact_time_hist(t_final, exact_num)

    #Set initial conditions
    #Delta_t and Storage (we will remember the whole history of y
    delta_t = t_final/n

    y_EE  = np.zeros((n+1, 1))
    y_IE  = np.zeros((n+1, 1))
    y_PC  = np.zeros((n+1, 1))
    time  = np.zeros((n+1, 1))

    y_EE[0] = exact(0.0)
    y_IE[0] = exact(0.0)
    y_PC[0] = exact(0.0)
    time[0] = 0.0

    # Start Time Marching Process
    for i in range(0, n):
        tn   = time[i]
        tnpo = tn + delta_t 
        time[i+1] = tnpo
        
        # Explicit and Implicit Euler Update
        y_EE[i+1] =  y_EE[i] + delta_t*(-0.5*y_EE[i]  + forcing(tn))
        y_IE[i+1] = (y_IE[i] + delta_t*forcing(tnpo)) / (1+delta_t/2)
        
        # Predictor Correcor MacCormack Update
        y_tempo   =  y_PC[i] +    delta_t*(-0.5*y_PC[i]  + forcing(tn))
        y_PC[i+1] =  y_PC[i] +0.5*delta_t*(-0.5*(y_tempo + y_PC[i]) + forcing(tn) +forcing(tnpo))
    
        # Errors
        error_ee = np.abs(exact(t_final) - y_EE[i+1])
        error_ie = np.abs(exact(t_final) - y_IE[i+1])
        error_pc = abs(exact(t_final) - y_PC[i+1])
  
    plt.figure(figsize=(10, 4))
    plt.title("\nTime Marching an ODE with Explicit Euler" + " >> Time Step:" + str(n))
    plt.plot(time, y_EE, marker='o', markersize=1, color= 'green', linestyle="dashed")
    plt.plot(exact_times, exact_sol, color = "blue")
    plt.legend(["Numerical Method", "Exact Sol"])
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.title("\n\nTime Marching an ODE with Implicit Euler" + " >> Time Step:" + str(n))
    plt.plot(time, y_IE, marker='o', markersize=1, color= 'green', linestyle="dashed")
    plt.plot(exact_times, exact_sol, color = "black")
    plt.legend(["Numerical Method", "Exact Sol"])
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.title("\n\nTime Marching an ODE with Predictor-Corrector" + " >> Time Step:" + str(n))
    plt.plot(time, y_PC, marker='o', markersize=1, color= 'green', linestyle="dashed")
    plt.plot(exact_times, exact_sol, color = "red")
    plt.legend(["Numerical Method", "Exact Sol"])
    plt.show()

if __name__ == '__main__':
    main()
# %%
