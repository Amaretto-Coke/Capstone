import os
import math
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
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


def make_cylinder_graphic(node_df, str_time_steps, inputs):
    times = []
    stills = []
    times = ['T @ ' + i for i in str_time_steps]
    timesteps = list(np.arange(0, (inputs['TimeIterations[#]'] + 1)))

    for n, i in enumerate(times):
        max_temp = node_df[i].max()
        min_temp = node_df[i].min()
        normalize = colors.Normalize(vmin=min_temp, vmax=max_temp)

        fig = plt.figure()
        ax = plt.gca()
        ax.set_aspect('equal')
        heatmap = ax.tricontourf(node_df['x'],
                                 node_df['y'],
                                 node_df[i],
                                 cmap=cm.viridis,
                                 norm=normalize,
                                 alpha=0.9)
        plt.axis('off')
        cbaxes = fig.add_axes([0.02, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        fig.colorbar(heatmap, cax=cbaxes, format='%.1f')

        new_axis = fig.add_axes(ax.get_position(),
                                projection='polar',
                                frameon=False,
                                rlabel_position=90)

        new_axis.set_theta_zero_location("S")
        new_axis.yaxis.grid(color='w', linewidth=0.75, alpha=0.2)
        new_axis.xaxis.grid(color='w', linewidth=0.75, alpha=0.2)
        radii_ticks = np.round(np.unique(node_df['radii'].values), 1)
        new_axis.set_rticks(radii_ticks)
        new_axis.set_title('Temperature After {0}s [K]'.format("{:2.2f}".format(timesteps[n] * inputs['TimeStep[s]'])))

        filename = os.path.dirname(os.getcwd()) + r'/tmp/3d_Temp_Step' + str(timesteps[n]) + '.png'
        stills.append(filename)
        plt.savefig(filename, dpi=96)
        plt.close(fig)

        print('\rCreating image {0} of {1}.'.format(
            int(n) + 1, len(times)),
            end='', flush=True)

    fps = len(timesteps) / 60

    print('\nCreating gif file...')
    gif_path = os.path.dirname(os.getcwd()) + r'/Output/TimeGraph.gif'
    with imageio.get_writer(gif_path, mode='I', duration=fps) as writer:
        for still in stills:
            image = imageio.imread(still)
            writer.append_data(image)

    print('\rCreating mp4 file...')
    vid_path = os.path.dirname(os.getcwd()) + r'/Output/TimeGraph.mp4'
    writer = imageio.get_writer(vid_path, fps=fps, macro_block_size=1)
    for still in stills:
        writer.append_data(imageio.imread(still))
    writer.close()


def outer_node_graph(node_df, time_steps, stop, str_time_steps):
    times = ['T @ ' + i for i in str_time_steps]
    temperature_df = node_df[['radii', 'theta']]

    for n, i in enumerate(times):
        temperature_df = pd.concat([temperature_df, node_df[i]], axis=1)

    # temperature_df = temperature_df[temperature_df.radii != temperature_df.radii.max()]
    outsidetemp = temperature_df[temperature_df.radii == temperature_df.radii.max()]
    temperature = outsidetemp.iloc[:, 2:]

    # setup the normalization and the colormap
    normalize = colors.Normalize(vmin=time_steps[0], vmax=time_steps[stop])
    colormap = cm.viridis
    fig = plt.figure()
    ax = plt.gca()

    for i, k in enumerate(temperature.iloc[0:, :stop]):
        plt.plot(outsidetemp.theta * (180 / np.pi), temperature[k], color=colormap(normalize(i)))

    # setup the colorbar and the figure
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(time_steps)
    plt.colorbar(scalarmappaple)
    plt.xticks([0, 90, 180, 270, 360])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xlabel("Theta (Degrees)")
    plt.ylabel("Temperature (K)")
    plt.title('Outer Node Temperature vs Theta for Different Time Steps')

    plt.show()


# make_cylinder_graphic(node_df,str_time_steps,inputs)
# outer_node_graph(node_df, time_steps, 5, str_time_steps)





