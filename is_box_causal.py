import numpy as np
import matplotlib.pyplot as plt

# Define the impulse response of the box filter
def box_filter_impulse_response(t):
    return np.where((t >= 0) & (t <= 1), 1, 0)

# Generate a time vector
t = np.linspace(-1, 2, 1000)

# Compute the impulse response
h = box_filter_impulse_response(t)

# Plot the impulse response
plt.plot(t, h)
plt.xlabel('Time (t)')
plt.ylabel('Impulse Response h(t)')
plt.title('Impulse Response of Box Filter')
plt.grid(True)
plt.show()

# Check if the impulse response is causal
is_causal = np.all(h[t < 0] == 0)

if is_causal:
    print("The box filter is causal.")
else:
    print("The box filter is not causal.")
