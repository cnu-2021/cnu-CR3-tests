import numpy as np
import matplotlib.pyplot as plt

def oscillator(w0, u0, v0, nmax, dt):
    '''
    Simulates the displacement of an oscillator with frequency w0,
    initial displacement u0 and velocity v0,
    using a finite difference method with step size dt,
    for nmax time steps.
    '''
    # Check that the step size is small enough
    if dt > 2 / w0:
        print(f'The step size is too large! For w0 = {w0}, choose dt < {2 / w0}.')
    elif dt == 2 / w0:
        print(f'The step size is at the stability limit, you may observe linear growth.')
    
    # Initialise an empty vector to store the result
    U = np.zeros(nmax)
    
    # Impose the initial conditions
    U[0] = u0
    U[1] = u0 + v0*dt

    # Compute the solution for the remaining nmax-2 time steps
    for n in range(2, nmax):
        U[n] = (2 - w0**2 * dt**2) * U[n-1] - U[n-2]
    
    return U


# Suggested test
w0 = 5
u0, v0 = 0.2, 5
nmax = 500
dt = 0.03

# Computed solution
U = oscillator(w0, u0, v0, nmax, dt)

# Exact solution
time = np.linspace(0, (nmax - 1) * dt, nmax)
u = u0 * np.cos(w0 * time) + v0 / w0 * np.sin(w0 * time)

fig, ax = plt.subplots()
ax.plot(time, u, 'k-', label='Exact')
ax.plot(time, U, 'r--', label='Simulated')
ax.set(xlabel='Time (s)', ylabel='Displacement')
ax.legend()
plt.show()


# Further tests

fig, ax = plt.subplots(1, 3)

# Stability limit: we should see linear growth
dt = 2 / w0
U = oscillator(w0, u0, v0, nmax, dt)

time = np.linspace(0, (nmax - 1) * dt, nmax)
u = u0 * np.cos(w0 * time) + v0 / w0 * np.sin(w0 * time)
ax[0].plot(time, u, 'k-', label='Exact')
ax[0].plot(time, U, 'r--', label='Simulated')
ax[0].set(xlabel='Time (s)', ylabel='Displacement', title='Stability limit')
ax[0].legend()

# Just over the stability limit: we should see exponential growth
dt = 1.0001 * (2 / w0)
U = oscillator(w0, u0, v0, nmax, dt)

time = np.linspace(0, (nmax - 1) * dt, nmax)
u = u0 * np.cos(w0 * time) + v0 / w0 * np.sin(w0 * time)
ax[1].plot(time, u, 'k-', label='Exact')
ax[1].plot(time, U, 'r--', label='Simulated')
ax[1].set(xlabel='Time (s)', ylabel='Displacement', title='Instability')
ax[1].legend()

# Just about stable: we should see large oscillations but no exponential growth
dt = 0.9999 * (2 / w0)
U = oscillator(w0, u0, v0, nmax, dt)

time = np.linspace(0, (nmax - 1) * dt, nmax)
u = u0 * np.cos(w0 * time) + v0 / w0 * np.sin(w0 * time)
ax[2].plot(time, u, 'k-', label='Exact')
ax[2].plot(time, U, 'r--', label='Simulated')
ax[2].set(xlabel='Time (s)', ylabel='Displacement', title='Close to instability')
ax[2].legend()

plt.show()


# Check that increasing w0 increases the frequency
fig, ax = plt.subplots()
tmax = 100

for w0 in range(1, 6, 2):
    dt = 0.1 * (2 / w0)
    nmax = int(tmax / dt)
    U = oscillator(w0, u0, v0, nmax, dt)

    time = np.linspace(0, (nmax - 1) * dt, nmax)
    ax.plot(time, U, label=f'w0 = {w0}')
    ax.set(xlabel='Time (s)', ylabel='Displacement')

    # Check that the first 2 values are set correctly
    print(U[0] == u0)
    print(U[1] == u0 + dt * v0)

ax.legend()
plt.show()


# Plot the difference between exact and computed solutions for different values of dt.
# Here the error should grow linearly for small dt (the first 2 values).
# The error is smaller for smaller dt (for small enough dt).
fig, ax = plt.subplots()
tmax = 500
w0 = 2

for dt_factor in [0.05, 0.1, 0.2]:
    dt = dt_factor * (2 / w0)
    nmax = int(tmax / dt)
    U = oscillator(w0, u0, v0, nmax, dt)

    time = np.linspace(0, (nmax - 1) * dt, nmax)
    u = u0 * np.cos(w0 * time) + v0 / w0 * np.sin(w0 * time)
    ax.plot(time, np.abs(U - u), label=f'dt = {dt:.2f}')
    ax.set(xlabel='Time (s)', ylabel='Absolute error')

ax.legend()
plt.show()

