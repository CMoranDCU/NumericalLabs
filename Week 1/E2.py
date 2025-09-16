import numpy as np
import matplotlib.pyplot as plt

# Define the function dx/dt = t - x^2
def f(t, x):
    return t - x**2

# Euler method implementation
def euler_method(f, x0, t0, t_end, dt):
    t_values = np.arange(t0, t_end, dt)
    x_values = np.zeros(len(t_values))
    x_values[0] = x0

    for i in range(1, len(t_values)):
        x_values[i] = x_values[i-1] + f(t_values[i-1], x_values[i-1]) * dt

    return t_values, x_values

# Parameters
x0 = 1  # Initial condition
t0 = 0  # Start time
t_end = 9  # End time
dt = 0.05  # Time step

# Run the Euler method
t_values, x_values = euler_method(f, x0, t0, t_end, dt)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_values, label='Euler Method', color='blue')
plt.title('Euler Method for dx/dt = t - x^2')
plt.xlabel('Time (t)')
plt.ylabel('x(t)')
plt.grid()
plt.legend()
plt.show()
