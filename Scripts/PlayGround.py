import numpy




import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def nx(r, x):
    return (r*x) * (1-x)


def plot_system(r, x0, n, ax=None):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(0, 1)
    ax.plot(t, nx(r, t), 'k', lw=2)
    ax.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    i=0
    for i in range(n):
        y = nx(r, x)
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'k', lw=1, alpha=(i + 1) / n)
        ax.plot([x, y], [y, y], 'k', lw=1, alpha=(i + 1) / n)
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=5, alpha=(i + 1) / n)
        x = y

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
plot_system(2.5, .1, 10, ax=ax1)
plot_system(3.5, .1, 100, ax=ax2)
plt.show()


n = 10000
r = np.linspace(2.8, 4.5, n)
c = np.arange(100)
c = 1/n

iterations = 10000
last = 200

x = 1e-5 * np.ones(n)

lyapunov = np.zeros(n)

gs = gridspec.GridSpec(6, 1)
mag = 1.2
fig = plt.figure(figsize=(7, 6))

ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])

for i in range(iterations):
    x = nx(r, x)
    # We compute the partial sum of the
    # Lyapunov exponent.
    lyapunov += np.log(abs(r - 2 * r * x))
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=0.25)

ax1.set_xlim(2.8, 4)
ax1.set_title("Bifurcation diagram")

# We display the Lyapunov exponent.
# Horizontal line.
ax2.axhline(0, color='k', lw=.5, alpha=.5)
# Negative Lyapunov exponent.
ax2.plot(r[lyapunov < 0],
         lyapunov[lyapunov < 0] / iterations,
         '.k', alpha=.5, ms=.5)

# Positive Lyapunov exponent.
ax2.plot(r[lyapunov >= 0],
         lyapunov[lyapunov >= 0] / iterations,
         '.r', alpha=.5, ms=.5)

ax2.set_xlim(2.8, 4)
ax2.set_ylim(-2, 1)
ax2.set_title("Lyapunov exponent")
plt.tight_layout()

plt.show()

print(type(1e-3))


'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(15, 2)
mag=1.2
fig = plt.figure(figsize=(8.5*mag, 11*mag))
fig.suptitle('ENPE 533 Assignment #1')

ax1 = fig.add_subplot(gs[:8, :])
ax2 = fig.add_subplot(gs[9:, 0])
ax3 = fig.add_subplot(gs[9:, 1])

axs = [ax1, ax2, ax3]


def graph(func_to_graph, form, var_range, ax, label='', color='b'):
    if form == 'y(x)':
        x = np.array(var_range)
        if label == '':
            ax.plot(x, func_to_graph(x), color=color)
        else:
            ax.plot(x, func_to_graph(x), color=color, label=label)
    elif form == 'x(y)':
        y = np.array(var_range)
        if label == '':
            ax.plot(func_to_graph(y), y, color=color)
        else:
            ax.plot(func_to_graph(y), y, color=color, label=label)
'''

'''
import pandas as pd

df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A4'],
                   'B': ['B0', 'B1', 'B2', 'B3', 'B4'],
                   'C': ['C0', 'C1', 'C2', 'C3', 'C4'],
                   'D': ['D0', 'D1', 'D2', 'D3', 'D4'],
                   't1': [2, 4, 8, 16, 32],
                   't2': [2, 4, 8, 17, 32],
                   'Z': [1, 1, 2, 3, 0]
                   })
print(df)

ne = (df['t1'] != df['t2']).any()

print(ne)

'''







