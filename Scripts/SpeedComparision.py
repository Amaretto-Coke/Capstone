import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

'''
def get_pkmn_pallet(hues=None, types=None):
    pkmn_colors = {
        'Bug':      {'light': 'C6D16E', 'regular': 'A8B820', 'dark': '6D7815'},
        'Dark':     {'light': 'A29288', 'regular': '705848', 'dark': '49392F'},
        'Dragon':   {'light': 'A27DFA', 'regular': '7038F8', 'dark': '4924A1'},
        'Electric': {'light': 'FAE078', 'regular': 'F8D030', 'dark': 'A1871F'},
        'Fairy':    {'light': 'F4BDC9', 'regular': 'EE99AC', 'dark': '9B6470'},
        'Fighting': {'light': 'D67873', 'regular': 'C03028', 'dark': '7D1F1A'},
        'Fire':     {'light': 'F5AC78', 'regular': 'F08030', 'dark': '9C531F'},
        'Flying':   {'light': 'C6B7F5', 'regular': 'A890F0', 'dark': '6D5E9C'},
        'Ghost':    {'light': 'A292BC', 'regular': '705898', 'dark': '493963'},
        'Grass':    {'light': 'A7DB8D', 'regular': '78C850', 'dark': '4E8234'},
        'Ground':   {'light': 'EBD69D', 'regular': 'E0C068', 'dark': '927D44'},
        'Ice':      {'light': 'BCE6E6', 'regular': '98D8D8', 'dark': '638D8D'},
        'Normal':   {'light': 'C6C6A7', 'regular': 'A8A878', 'dark': '6D6D4E'},
        'Poison':   {'light': 'C183C1', 'regular': 'A040A0', 'dark': '682A68'},
        'Psychic':  {'light': 'FA92B2', 'regular': 'F85888', 'dark': 'A13959'},
        'Rock':     {'light': 'D1C17D', 'regular': 'B8A038', 'dark': '786824'},
        'Steel':    {'light': 'D1D1E0', 'regular': 'B8B8D0', 'dark': '787887'},
        'Water':    {'light': '9DB7F5', 'regular': '6890F0', 'dark': '445E9C'},
        '???':      {'light': '9DC1B7', 'regular': '68A090', 'dark': '44685E'}
    }

    if hues is not None and types is not None:
        # If the func call specifies both the type(s) and the hues
        result = pkmn_colors
        pop_list = []
        for key in result.keys():
            if key not in types:
                pop_list.append(key)
        for key in pop_list:
            result.pop(key)
        for type_hue in result:
            x = result[type_hue]
            pop_list = []
            for y in x:
                if y not in hues:
                    pop_list.append(y)
            for y in pop_list:
                x.pop(y)
            result.pop(type_hue)
            result[type_hue] = x
        print(result)
    elif hues is not None:
        # If the func call specifies the hues but not the type
        pass
    elif types is not None:
        pass
    else:
        pass
'''

# get_pkmn_pallet(hues=['light'], types=['Water', 'Dark'])


def normalize_vector(vector):
    """
    Takes a vector and returns its' unit vector.
    :param vector: The vector to be normalized, as a 1 x 3 numpy array.
    :return: The normalized vector, as a 1 x 3 numpy array.
    """
    to_sum = []
    for p in vector:
        to_sum.append(p**2)
    vector_mag = math.sqrt(sum(to_sum))
    return vector/vector_mag


def make_my_plot(df, xyz_var_names, title, colors, math_str='', scatter_label=''):

    X = []
    Y = []

    for index, row in df.iterrows():
        y, x1, x2 = row[xyz_var_names['z']], row[xyz_var_names['x']], row[xyz_var_names['y']]
        X.append([float(x1), float(x2), 1])  # add the bias term at the end
        Y.append(float(y))

    # use numpy arrays so that we can use linear algebra later
    X = np.array(X)
    Y = np.array(Y)

    # Use Linear Algebra to solve
    a = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    predictedY = np.dot(X, a)

    # calculate the r-squared
    SSres = Y - predictedY
    SStot = Y - Y.mean()
    rSquared = 1 - (SSres.dot(SSres) / SStot.dot(SStot))

    R_str = 'R² = ' + '{:.4f}'.format(rSquared) + '.'

    a_str = normalize_vector(a)
    a_str = ['{:.4f}'.format(coff) for coff in a_str]
    unit_vect_str = 'The Unit Vectors for ' + xyz_var_names['x'] + \
                    ', ' + xyz_var_names['y'] + ', and ' + \
                    xyz_var_names['z'] + ' are:'
    for coff in a_str:
        unit_vect_str += ' ' + coff
    unit_vect_str += '.'

    eqtn_str = 'The Equation of the Plane is:\n'+ a_str[0] + '∙X '
    if float(a_str[1]) >= 0:
        eqtn_str += '+ ' + a_str[1] + '∙Y '
    else:
        eqtn_str += '- ' + str(abs(float(a_str[1]))) + '∙Y '

    if float(a_str[2]) >= 0:
        eqtn_str += '+ ' + a_str[2] + '∙Z '
    else:
        eqtn_str += '- ' + str(abs(float(a_str[2]))) + '∙Z '

    if xyz_var_names['z'] == 'Old Time/New Time':
        eqtn_str += '+ 1 = 1'
    else:
        eqtn_str += '= 1'

    # create a wiremesh for the plane that the predicted values will lie
    xx, yy, zz = np.meshgrid(X[:, 0], X[:, 1], X[:, 2])
    combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    Z = combinedArrays.dot(a)

    mag = 1.25  # magnify the figure
    fig = plt.figure(figsize=(6.4*mag, 4.8*mag))

    ax = fig.gca(projection='3d')

    scatter = ax.scatter(X[:, 0], X[:, 1], Y, color=colors['dots'])
    if scatter_label != '':
        scatter.set_label(scatter_label)
    ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5, color=colors['surf'])

    ax.set_zlim(bottom=0)
    ax.set_facecolor('w')
    ax.set_xlabel(xyz_var_names['x'] + ' [X]')
    ax.set_ylabel(xyz_var_names['y'] + ' [Y]')
    ax.set_zlabel(xyz_var_names['z'] + ' [Z]')
    if scatter_label != '':
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.13))

    plt.title(title, y=1.15)
    plt.subplots_adjust(bottom=0.18, top=0.83)
    plt.gcf().text(0.5, 0.025, eqtn_str, fontsize=9, ha='center')
    plt.gcf().text(0.5, 0.075, R_str, fontsize=9, ha='center')
    if math_str != '':
        plt.gcf().text(0.95, .07, math_str, fontsize=14, ha='right')
    plt.show()

path = os.path.dirname(os.getcwd()) + '\Output\SpeedOut.txt'

data = pd.read_csv(path, sep=' ')

data = data.groupby(by=['Script', 'Nodes', 'Iterations']).mean().reset_index()

old = data[data['Script'] == 'Old'].copy(deep=True)
new = data[data['Script'] == 'New'].copy(deep=True)

new['Old Time/New Time'] = 1
old['Old Time/New Time'] = old['Seconds'].to_numpy() / new['Seconds'].to_numpy()

#  Making the Old Time Plot
title_str = 'Old Iteration Time'
plot_colors = {'dots': 'r', 'surf': 'm'}
pnt_eq_str = r'$Z_{i,j} = \sum_{k=0}^n t_{Old,i,j,k}$'
var_names = {'x': 'Nodes', 'y': 'Iterations', 'z': 'Seconds'}
make_my_plot(df=old, xyz_var_names=var_names, colors=plot_colors, title=title_str, math_str=pnt_eq_str)

#  Making the New Time Plot
title_str = 'New Iteration Time'
plot_colors = {'dots': 'b', 'surf': 'c'}
pnt_eq_str = r'$Z_{i,j} = \sum_{k=0}^n t_{New,i,j,k}$'
var_names = {'x': 'Nodes', 'y': 'Iterations', 'z': 'Seconds'}
make_my_plot(df=new, xyz_var_names=var_names, colors=plot_colors, title=title_str, math_str=pnt_eq_str)

#  Making the Old vs New Time Comparision
title_str = 'Old Iteration Time Normalized to New Iteration Time.'
plot_colors = {'dots': 'k', 'surf': 'gray'}
var_names = {'x': 'Nodes', 'y': 'Iterations', 'z': 'Old Time/New Time'}
pnt_eq_str = r'$Z_{i,j} = \frac{\sum_{k=0}^n t_{Old,i,j,k}}{\sum_{k=0}^n t_{New,i,j,k}}$'
scatter_name = 'Relative Speed of New Function to Old'
make_my_plot(df=old, xyz_var_names=var_names, colors=plot_colors, title=title_str, math_str=pnt_eq_str, scatter_label=scatter_name)

print("Press Enter to continue ...")
input()
