import os
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors



node_df = pd.read_csv(os.path.dirname(os.getcwd()) + r'\output_df.csv')

max_temp = node_df['Temp'].max()
min_temp = node_df['Temp'].min()
normalize = colors.Normalize(vmin=min_temp, vmax=max_temp)

fig = plt.figure()
ax = plt.gca()
ax.set_aspect('equal')
heatmap = ax.tricontour(node_df['x'],
                        node_df['y'],
                        node_df['Temp'],
                        cmap=cm.rainbow,
                        norm=normalize,
                        alpha=0.9)
plt.axis('off')
cbaxes = fig.add_axes([0.02, 0.1, 0.03, 0.8])  # This is the position for the colorbar
cb = fig.colorbar(heatmap, cax=cbaxes, format='%.1f')
for line in cb.lines:
   line.set_linewidth(35)

new_axis = fig.add_axes(ax.get_position(),
                        projection='polar',
                        frameon=False,
                        rlabel_position=90)

new_axis.yaxis.grid(color='k', linewidth=0.75, alpha=1)
new_axis.xaxis.grid(False)
new_axis.set_xticks([])
new_axis.set_rticks([1, 2, 4, 6])
new_axis.tick_params(labelleft=False)


"""
#  Figuring out which solution to the equation a*x**4 + b*x = c
#  https://www.wolframalpha.com/input/?i=ax%5E4%2Bbx%3Dc

#  Testing the speed of the various root functions

import timeit
import itertools

mysetup1 = '''
import itertools

def sqrt(num):
    return num ** 0.5


def root1(a, b, c):
    return 1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt(-(2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root2(a, b, c):
    return 1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))+1/2*sqrt(-(2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root3(a, b, c):
    return -1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt((2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root4(a, b, c):
    return 1/2*sqrt((2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


a = [1, 10, 100]
b = [1, 10, 100]
c = [1, 10, 100]

abc = [a, b, c]

combos = list(itertools.product(*abc))
'''

mysetup2 = '''
import itertools

def sqrt(num):
    return -num ** 0.5


def root1(a, b, c):
    return 1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt(-(2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root2(a, b, c):
    return 1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))+1/2*sqrt(-(2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root3(a, b, c):
    return -1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt((2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root4(a, b, c):
    return 1/2*sqrt((2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


a = [1, 10, 100]
b = [1, 10, 100]
c = [1, 10, 100]

abc = [a, b, c]

combos = list(itertools.product(*abc))
'''

mycode1 = '''
for com in combos:
    r1 = root1(com[0], com[1], com[2])
'''
mycode2 = '''
for com in combos:
    r2 = root2(com[0], com[1], com[2])
'''
mycode3 = '''
for com in combos:
    r3 = root3(com[0], com[1], com[2])
'''
mycode4 = '''
for com in combos:
    r4 = root4(com[0], com[1], com[2])
'''

mycodelist = [mycode1, mycode2, mycode3, mycode4]
x = 0
while x <= 15:
    break
    print("\n'+SQRT")
    for mycode in mycodelist:
        print('{:.3f}'.format(
            timeit.timeit(setup=mysetup1,
                            stmt=mycode,
                            number=1000)
        )
              )

    print("'-SQRT")
    for mycode in mycodelist:
        pass
        print('{:.3f}'.format(
            timeit.timeit(setup=mysetup2,
                            stmt=mycode,
                            number=1000)
        )
              )

    x += 1


#  Testing the root equations for real and positive results with positive coefficients

import itertools
import timeit


def sqrt(num):
    return num **0.5


def root1(a, b, c):
    return 1/2*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5-1/2*(-(2*b)/(a*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5)-(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5


def root2(a, b, c):
    return 1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))+1/2*sqrt(-(2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root3(a, b, c):
    return -1/2*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))-1/2*sqrt((2*b)/(a*sqrt((sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)))-(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(sqrt(3)*sqrt(256*a**3*c**3+27*a**2*b**4)+9*a*b**2)**(1/3))


def root4(a, b, c):
    return 1/2*((2*b)/(a*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5)-(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5-1/2*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5


a = [1, 10, 100]
b = [1, 10, 100]
c = [1, 10, 100]

abc = [a, b, c]

combos = list(itertools.product(*abc))

for com in combos:
    r1 = root1(com[0], com[1], com[2])
    r2 = root2(com[0], com[1], com[2])
    r3 = root3(com[0], com[1], com[2])
    r4 = root4(com[0], com[1], com[2])
    print(com, '\t', '\t',
          '({0.real:.2f} {0.imag:+.2f}j)'.format(r1), '\t',
          '({0.real:.2f} {0.imag:+.2f}j)'.format(r2), '\t',
          '({0.real:.2f} {0.imag:+.2f}j)'.format(r3), '\t',
          '({0.real:.2f} {0.imag:+.2f}j)'.format(r4))
          


#  Simplifying the root function

str_root4 = '''
def root4(a, b, c):
    return 1/2*((2*b)/(a*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5)-(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5-1/2*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5'''


def longestRepeatedSubstring(str):
    n = len(str)
    LCSRe = [[0 for x in range(n + 1)]
             for y in range(n + 1)]

    res = ""  # To store result
    res_length = 0  # To store length of result

    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):

            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (str[i - 1] == str[j - 1] and
                    LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1

                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)

            else:
                LCSRe[i][j] = 0

    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                       index + 1):
            res = res + str[i - 1]

    return res

#  Simplifying the equation:
print(str_root4)
# Out:
'''
def root4(a, b, c):
    return 1/2*((2*b)/(a*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5)-(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5-1/2*((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5
'''

tmp = longestRepeatedSubstring(str_root4)
print('tmp:', tmp)
# Out:
# tmp: *((3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/(3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3))**0.5
A = longestRepeatedSubstring(tmp)
print('A:', A)
# Out:
# A: (3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)


str_root4_with_A = str_root4.replace(A, 'A')
print('\n', str_root4_with_A)
# Out:
'''
def root4(a, b, c):
    return 1/2*((2*b)/(a*(A/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/A)**0.5)-A/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/A)**0.5-1/2*(A/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/A)**0.5
'''

B = longestRepeatedSubstring(str_root4_with_A)
print('B:', B)
# Out:
# B: *(A/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/A)**0.5

str_root4_with_AB = str_root4_with_A.replace(B, '*B')
print('\n', str_root4_with_AB)
# Result:
'''
def root4(a, b, c):
    return 1/2*((2*b)/(a*B)-A/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/A)**0.5-1/2*B
'''


import numpy as np


def root4improved(a, b, c):
    A = (3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)
    B = (A/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/A)**0.5
    C = 1/2*((2*b)/(a*B)-A/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/A)**0.5-1/2*B
    return np.where(a != 0, np.real(C), np.real(c/b))


my_a = np.array([0, 10, 0], dtype=complex)
my_b = np.array([3, 100, 7], dtype=complex)  # B arguments cannot equal 0
my_c = np.array([2, 0, 10], dtype=complex)

res = root4improved(my_a, my_b, my_c)

print(res)
print(res)

"""

'''
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







