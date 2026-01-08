import numpy as np
import matplotlib.pyplot as plt

# Define linear frequency range from -20 to 20
w = np.linspace(-20, 20, 1000)
s = 1j * w

# H(jw) = 2(2 + jw) / ((1 + jw)(3 + jw))
H = (2 * (2 + s)) / ((1 + s) * (3 + s))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Phase Plot (Top)
ax1.plot(w, np.angle(H, deg=True), color='tab:blue', linewidth=2)
ax1.set_title(r'Phase Response of $H(j\omega)$')
ax1.set_ylabel(r'$\angle H(j\omega)$ (degrees)')
ax1.set_xlabel(r'$\omega$')
ax1.set_ylim([-100, 100])
ax1.set_xlim([-20, 20])
ax1.grid(True)

# Magnitude Plot (Bottom)
ax2.plot(w, np.abs(H), color='tab:blue', linewidth=2)
ax2.set_title(r'Magnitude Response of $H(j\omega)$')
ax2.set_ylabel(r'$|H(j\omega)|$')
ax2.set_xlabel(r'$\omega$')
ax2.set_ylim([0, 1.4])
ax2.set_xlim([-20, 20])
ax2.grid(True)

plt.tight_layout()
plt.savefig('homework_plots.png')
plt.show()