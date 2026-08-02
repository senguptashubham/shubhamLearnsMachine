import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags


delta = 2

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax_signals, ax_corr, ax_stretch, ax_stretch_corr = axes.flat

sig_1 = np.array([np.sin(x) for x in range(0,50)])
ax_signals.plot(sig_1, label='sin(x)')
sig_2 = np.array([np.sin(x-delta) for x in range(0,50)])
ax_signals.plot(sig_2, label=f'sin(x-{delta})')
ax_signals.legend()

abs_diff = np.abs(sig_1 - sig_2) #difference calculated for point pair (0,0) to (49,49)
mean_diff = np.mean(abs_diff)
print(f"Mean Defference is {mean_diff} while absolute difference is {abs_diff}")

def plot_lags_correlation(sig1, sig2, subplot):
  corr = correlate(sig2, sig1, mode='full')
  lags = correlation_lags(len(sig2), len(sig1), mode='full')
  lag = lags[np.argmax(corr)]
  subplot.plot(lags, corr)
  subplot.set_xlabel('Lags')
  subplot.set_ylabel('Correlation')
  return lag, subplot


lag_corr, ax_corr = plot_lags_correlation(sig_1, sig_2, ax_corr)
print(f"estimated lag: {lag_corr}")

N, a = 200, 0.15
t = np.linspace(0, 1, N)
sig_3 = np.sin(2 * np.pi * t)
w = t + a * (np.sin(2 * np.pi * t))
sig_4 = np.sin(2 * np.pi * w)
ax_stretch.plot(sig_3, label='original')
ax_stretch.plot(sig_4, label='locally stretched')

lag_stretch, ax_stretch_corr = plot_lags_correlation(sig_3, sig_4, ax_stretch_corr)

fig.tight_layout()
plt.show()