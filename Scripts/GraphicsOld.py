import os
import math
import imageio
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


def color_nodes_by_component(comp):

    if comp == 'Liquid':

        color = 'b'  # b for blue

    elif comp == 'Gas':

        color = 'c'  # c for cyan

    else:

        color = 'k'  # k for black

    return color





def multiple_formatter(denominator=2, number=np.pi, latex=r'\pi'):

    def gcd(a, b):

        while b:

            a, b = b, a % b

        return a



    def _multiple_formatter(x, pos):

        den = denominator

        num = np.int(np.rint(den*x/number))

        com = gcd(num, den)

        (num, den) = (int(num/com), int(den/com))

        if den == 1:

            if num == 0:

                return r'$0$'

            if num == 1:

                return r'$%s$' % latex

            elif num == -1:

                return r'$-%s$' % latex

            else:

                return r'$%s%s$' % (num, latex)

        else:

            if num == 1:

                return r'$\frac{%s}{%s}$' % (latex, den)

            elif num == -1:

                return r'$\frac{-%s}{%s}$' % (latex, den)

            else:

                return r'$\frac{%s%s}{%s}$' % (num, latex, den)

    return _multiple_formatter





class Multiple:

    def __init__(self, denominator=2, number=np.pi, latex=r'\pi'):

        self.denominator = denominator

        self.number = number

        self.latex = latex



    def locator(self):

        return plt.MultipleLocator(self.number / self.denominator)



    def formatter(self):

        return plt.FuncFormatter(multiple_formatter(self.denominator, self.number, self.latex))





def generate_3d_node_geometry(prop_df):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(prop_df.x.to_list(),

               prop_df.y.to_list(),

               prop_df.z.to_list(),

               c=prop_df.c.to_list(),

               s=5)

    plt.show()





def generate_boundary_graphs(temp_df, prop_df, time_steps, features, labels, color_map='hsv'):

    copy_df = temp_df.copy(deep=True)

    copy_df.reset_index(inplace=True)

    copy_df.set_index('NodeIdx',

                      drop=False,

                      inplace=True)

    copy_df = copy_df.merge(prop_df[['theta', 'radii']],

                            left_index=True,

                            right_index=True)

    copy_df = copy_df[copy_df['theta'] > 0]

    copy_df = copy_df[copy_df['radii'] == copy_df['radii'].max()]



    fig, axs = plt.subplots(nrows=len(features),

                            ncols=1,

                            sharex=True,

                            figsize=(10, 20))

    palette = plt.get_cmap(color_map)



    for ax, feature, label in zip(axs.flatten(), features, labels):

        lines_df = copy_df.copy(deep=True)

        for time in range(0, len(time_steps) - 1):

            line_df = lines_df[lines_df['TimeStep'] == time_steps.iloc[time]]

            ax.plot(line_df['theta'],

                    line_df[feature],

                    label=time_steps.iloc[time],

                    color=palette(time),

                    linewidth=1,

                    alpha=0.9)



        ax.grid(True)

        ax.set_xlim(left=0, right=2 * math.pi)

        ax.set_ylabel(label)



        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))

        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))

        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    plt.xlabel('Phase Angle')

    plt.subplots_adjust(hspace=0.0, left=0.15)

    plt.show()





def generate_time_gif(temp_df, prop_df, time_steps):

    copy_df = temp_df.copy(deep=True)

    copy_df.reset_index(inplace=True)

    copy_df.set_index('NodeIdx', drop=False, inplace=True)

    copy_df = copy_df.merge(prop_df[['x', 'y', 'z']], left_index=True, right_index=True)

    time_steps = time_steps.to_list()

    cmap = cm.viridis

    max_temp = copy_df['Temp[K]'].max()

    min_temp = copy_df['Temp[K]'].min()

    normalize = colors.Normalize(vmin=min_temp, vmax=max_temp)

    stills = []

    fig = plt.figure()

    for time in time_steps:



        # Make the plot

        fig = plt.figure()

        ax = fig.gca(projection='3d')

        point_df = copy_df[copy_df['TimeStep'] == time]

        s = ax.scatter(point_df['x'],

                       point_df['y'],

                       point_df['z'],

                       c=point_df['Temp[K]'],

                       cmap=cmap,

                       norm=normalize,

                       alpha=0.9)

        ax.set_title('Temperature After {0}s [K]'.format("{:2.2f}".format(time)))

        fig.colorbar(s)



        filename = os.path.dirname(os.getcwd()) + r'\tmp\3d_Temp_Step' + str(time_steps.index(time)) + '.png'

        stills.append(filename)

        plt.savefig(filename, dpi=96)

        plt.gca()

        plt.close(fig)



    fps = len(time_steps)/30



    gif_path = os.path.dirname(os.getcwd()) + r'\Output\TimeGraph.gif'

    with imageio.get_writer(gif_path, mode='I', duration=fps) as writer:

        for still in stills:

            image = imageio.imread(still)

            writer.append_data(image)



    vid_path = os.path.dirname(os.getcwd()) + r'\Output\TimeGraph.mp4'

    writer = imageio.get_writer(vid_path, fps=fps)

    for still in stills:

        writer.append_data(imageio.imread(still))

    writer.close()

