import numpy as np
import matplotlib.pyplot as plt

# Configuration
initial_valence = 50
decay_rate = 0.01
valence_floor = 5
max_steps = 100

# Compute decaying valence over steps
steps = np.arange(max_steps + 1)
valence_values = np.maximum(initial_valence * np.exp(-decay_rate * steps), valence_floor)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(steps, valence_values, marker='o')
plt.title('Fading Valence over Steps (Exponential Decay)')
plt.xlabel('Step')
plt.ylabel('Valence')
plt.grid(True)
plt.tight_layout()
plt.show()
