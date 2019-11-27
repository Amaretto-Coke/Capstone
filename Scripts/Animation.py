import numpy as np

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation

plt.style.use('seaborn-pastel')

fig = plt.figure()
ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
line = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line


def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,


anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

anim.save('sine_wave.gif', writer='imagemagick')


'''
import math
from decimal import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def beta_pdf(x, a, b):
    c = x**(a-1)
    d = (1-x)**(b-1)
    e = math.factorial(a + b - 1)
    f = (math.factorial(a - 1) * math.factorial(b - 1))
    g = (c * d)
    h = (e // f)
    try:
        return g * h
    except OverflowError:
        plt.close()
        plt.show()
        return


class UpdateDist(object):
    def __init__(self, ax, prob=0.5):
        self.success = 0
        self.prob = prob
        self.line, = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 25)
        self.ax.grid(True)


        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        self.ax.axvline(prob, linestyle='--', color='black')

    def init(self):
        self.success = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            return self.init()

        # Choose success based on exceed a threshold with a uniform pick
        if np.random.rand(1,) < self.prob:
            self.success += 1
        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)
        return self.line,

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()
ud = UpdateDist(ax, prob=.76)
anim = FuncAnimation(fig, ud, frames=range(1000), init_func=ud.init,
                     interval=25, blit=True)
plt.show()
'''

