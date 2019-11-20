import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
# import scipy as sp
# import os
from PostOffice import *
# from scipy.integrate import quad
# from sklearn import preprocessing

''' 
Testing Data Export Function
data = [['chris', 10], ['nick', 15], ['juli', 14]]

df1 = pd.DataFrame(data=data)
df2 = pd.DataFrame(data=data)

dfs = [df1, df2]

export_results(dfs, ['df1', 'df2'])

path = os.path.dirname(os.getcwd()) + r'\MailBox.xlsx'

os.startfile(path)

'''


def get_df_value(df_to_search, idx0, idx1, idx2, column):
    """
    A function to retrieve values from a triple indexed dataframe.
    This function uses a circular approach to indexing, for example
        an index of 6 when the dataframe only contains 4 rows will
        produce the second entry.

    :param df_to_search: The dateframe that the value is to be retrieved from.
    :param idx0: The first index, appearing at the top level.
    :param idx1: The second index, found within the first index.
    :param idx2: The third index, found within the second index
    :param column:  The column from the indexed row that contains
                        the value to be retrieved.
    :return:    The return will be whatever value is stored in the dataframe
                    column specified at the indexed location.
                The type of the return depends on the type of the value
                    stored in the dataframe.
    """

    df_idx0 = sorted(list(set(df_to_search.index.get_level_values(0))))
    df_idx1 = sorted(list(set(df_to_search.index.get_level_values(1))))
    df_idx2 = sorted(list(set(df_to_search.index.get_level_values(2))))

    if idx0 in df_idx0:
        pass
    elif idx0 > max(df_idx0):
        idx0 -= max(df_idx0) + 1
    elif idx0 < min(df_idx0):
        idx0 += max(df_idx0) - 1
    else:
        print('Level', idx0, 'not in, above, or below df levels.')

    if idx1 in df_idx1:
        pass
    elif idx1 > max(df_idx1):
        idx1 -= max(df_idx1) + 1
    elif idx1 < min(df_idx1):
        idx1 += max(df_idx1) - 1
    else:
        print('Column', idx1, 'not in, above, or below df columns.')

    if idx2 in df_idx2:
        pass
    elif idx2 > max(df_idx2):
        idx2 -= max(df_idx2) + 1
    elif idx2 < min(df_idx2):
        idx2 += max(df_idx2) - 1
    else:
        print('Row', idx2, 'not in, above, or below df rows.')

    return df_to_search.loc[(idx0, idx1, idx2)][column]


def vf_plane_to_cyl1(s, r, l, t, n):
    slices = list(range(0, n+1))
    slices = [t/2 * i/max(slices) for i in slices]
    points = []
    areas = []

    for x in slices:
        a = s**2 + x**2
        b = l**2 - s**2 - x**2 + r**2
        c = l**2 + s**2 + x**2 - r**2
        d = math.sqrt(c**2 + 4 * l**2 * r**2)
        e = d * math.acos((r * b)/(math.sqrt(a) * c)) + \
            b * math.asin(r/math.sqrt(a)) - \
            math.pi * c / 2
        g = math.acos(b/c) - e / (2 * r * l)
        f = s * r / a * (1 - g / math.pi)
        points.append(f)

    for i in range(0, n):
        areas.append((slices[i + 1] - slices[i]) *
                     (points[i] + points[i + 1]) / 2)

    return sum(areas) * 2 / t


def vf_plane_to_cyl2(s, r, l, t, n):
    R = r/l
    Z = s/r
    T = t/r

    slices = list(range(0, n + 1))
    slices = [t/2 * i / max(slices) for i in slices]
    # equation calls for '1' in the place of the 't/2' in the line above,
    points = []
    areas = []

    for x in slices:
        y = R**2 * (1 - Z**2 - T**2 * x**2 / 4)
        v = (Z**2 + T**2 * x**2 / 4)**(-1/2)
        a = math.sqrt((1-y)**2 + 4 * R**2)
        b = v * (1 + y) / (1 - y)
        c = (a * math.acos(b) +
             (1 + y) * math.asin(v) -
             math.pi * (1 - y) / 2)
        d = (math.acos((1 + y) / (1 - y)) -
             c / (2 * R))
        f = Z * v**2 * (1 - d / math.pi)
        points.append(f)

    for i in range(0, n):
        area = (slices[i + 1] - slices[i]) * (points[i + 1] + points[i]) / 2
        areas.append(area)

    return sum(areas) * T / (2 * math.pi)


def vf_plane_to_plane(s, l, t):
    x = l/s
    y = t/s
    a = math.log(
        (1 + x**2) * (1 + y**2) / (1 + x**2 + y**2)
    )**(1/2)
    b = x * (1 + y**2)**(1/2) * math.atan(x / (1 + y**2)**(1/2))
    c = y * (1 + x**2)**(1/2) * math.atan(y / (1 + x**2)**(1/2))
    d = x * math.atan(x) + y * math.atan(y)
    return 2 * (a + b + c - d) / (math.pi * x * y)


def points_all_on_plane(p1, p2, p3, p4):
    """
    Evaluates 4 coordinates to determine if they are on the
        same plane in 3D space.
    :param p1: The first    point, as a 1 x 3 numpy array.
    :param p2: The second   point, as a 1 x 3 numpy array.
    :param p3: The third    point, as a 1 x 3 numpy array.
    :param p4: The forth    point, as a 1 x 3 numpy array.
    :return: True or False, depending on if all 4 points are on a plane.
    """

    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1

    cp1 = np.cross(v1, v2)
    cp2 = np.cross(v1, v3)

    cp1 = normalize_vector(cp1)
    cp2 = normalize_vector(cp2)

    result = (cp1[0] == cp2[0]) and (cp1[1] == cp2[1]) and (cp1[2] == cp2[2])

    return result


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


def create_layer_nodes(slices=1,
                       rings=1,
                       space_out=False,
                       vol_factor=1.2):

    delta_theta = 2 * math.pi / slices
    delta_radii = 1/rings

    lev_nodes = []
    radii = delta_radii / 2
    vol_limit = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta / 2 * vol_factor
    # The volume of the innermost ring * 1.2 is the the maximum volume that an element in a new ring may have.
    # Otherwise, the number of elements in that ring get doubled.

    while radii < 1:
        e_vol = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta/2
        if space_out:
            while e_vol > vol_limit:
                delta_theta /= 2
                e_vol = ((radii + delta_radii / 2) ** 2 - (radii - delta_radii / 2) ** 2) * delta_theta / 2
        theta = delta_theta/2
        while theta < 2 * math.pi:
            lev_nodes.append([theta, radii, delta_theta])
            theta += delta_theta
        radii += delta_radii

    df = pd.DataFrame(data=np.asarray(lev_nodes), columns=['theta', 'radii', 'delta_theta'])

    df['delta_radii'] = delta_radii

    return df


def generate_xyzn(df, base_center=None):

    if base_center is None:
        base_center = [0, 0, 0]

    df['x'] = base_center[0] + \
              df['radii'] * \
              df['theta'].apply(lambda theta: math.cos(theta))
    df['y'] = base_center[1] + \
              df['radii'] * \
              df['theta'].apply(lambda theta: math.sin(theta))
    df['z'] = base_center[2] + df['height']
    df['n'] = np.empty((len(df), 0)).tolist()
    for i in range(0, len(df.index)):
        delta_x = df.at[i, 'x'] - base_center[0]
        delta_y = df.at[i, 'y'] - base_center[1]
        delta_z = df.at[i, 'height'] + base_center[2]
        df.at[i, 'n'] = [delta_x, delta_y, delta_z]

    return df


def create_partial_cyl(layer_df,
                       layers=1,
                       cyl_height=1.0,
                       base_center=None,
                       component=None):

    delta_height = (cyl_height - base_center[2])/layers
    height = delta_height/2 + base_center[2]
    cyl_df = pd.DataFrame(columns=['theta', 'radii', 'delta_theta', 'height'])
    my_layer_df = pd.DataFrame(columns=['theta', 'radii', 'delta_theta'])

    while height < cyl_height:
        my_layer_df = layer_df.copy()
        my_layer_df['height'] = height
        cyl_df = cyl_df.append(my_layer_df, ignore_index=True, sort=False)
        height += delta_height

    cyl_df['component'] = component
    del my_layer_df
    cyl_df.reset_index(inplace=True, drop=True)

    cyl_df['delta_height'] = delta_height

    # cyl_df = generate_xyzn(cyl_df, base_center=base_center)

    return cyl_df


def create_cyl_wall(df, wall_thickness=None):
    if df is None:
        print('No data frame to create cylinder wall from.')
        quit()
    if wall_thickness is None:
        wall_thickness = df['delta_radii'].max()
    outer_cyl_nodes = df[df['radii'] == df['radii'].max()]
    outer_cyl_nodes['radii'] = outer_cyl_nodes['radii'] + wall_thickness
    outer_cyl_nodes.component = 'Wall'
    df = df.append(outer_cyl_nodes, ignore_index=True, sort=False)
    df.reset_index(inplace=True, drop=True)
    return df


def create_cyl_wall(df, wall_thickness=None):
    if df is None:
        print('No data frame to create cylinder wall from.')
        quit()
    if wall_thickness is None:
        wall_thickness = df['delta_radii'].max()
    outer_cyl_nodes = df.loc[df['radii'] == df['radii'].max()].copy()
    outer_cyl_nodes['radii'] = outer_cyl_nodes['radii'] + wall_thickness
    outer_cyl_nodes.component = 'Wall'
    df = df.append(outer_cyl_nodes, ignore_index=True, sort=False)
    df.reset_index(inplace=True, drop=True)
    return df


def assign_neighbors(df):
    df['lft_nbr'] = 0
    df['rht_nbr'] = 0
    df['inr_nbr'] = 0
    df['otr_nbr'] = pd.np.empty((len(df), 0)).tolist()

    # Finding the left, right, and inner neighbors.
    for node in range(0, len(df)):
        # finding the left neighbor
        my_df1 = df[df['radii'] == df.loc[node]['radii']]
        my_df2 = my_df1[my_df1['theta'] > df.loc[node]['theta']]
        if len(my_df2) != 0:
            nbr = my_df2[['theta']].idxmin()
        else:
            nbr = my_df1[['theta']].idxmin()
        df.at[node, 'lft_nbr'] = nbr

        # finding the right neighbor
        my_df2 = my_df1[my_df1['theta'] < df.loc[node]['theta']]
        if len(my_df2) != 0:
            nbr = my_df2[['theta']].idxmax()
        else:
            nbr = my_df1[['theta']].idxmax()
        df.at[node, 'rht_nbr'] = nbr

        # finding the inner neighbor
        my_df1 = df[df['radii'] < df.loc[node]['radii']]
        if len(my_df1) != 0:
            my_df1 = my_df1[my_df1['radii'] == my_df1['radii'].max()]
            my_df1['theta_diff'] = abs(my_df1['theta'] - df.loc[node]['theta'])
            nbr = my_df1[['theta_diff']].idxmin()
        else:
            nbr = None
        df.at[node, 'inr_nbr'] = nbr

    df['inr_nbr'] = df['inr_nbr'].astype('Int64')

    # finding the outer neighbors
    for node in range(0, len(df)):
        my_df1 = df[df['inr_nbr'] == node]
        my_df1 = my_df1[my_df1['height'] == df.loc[node]['height']]
        if len(my_df1) != 0:
            nbr = [i for i in list(my_df1.index.values)]
        else:
            nbr = []
        df.at[node, 'otr_nbr'] = nbr

    return df


def create_cyl_nodes(slices=1,
                     rings=1,
                     gas_layers=1,
                     liq_layers=1,
                     cyl_diam=1.0,
                     cyl_height=1.0,
                     liq_level=0.5,
                     base_center=None,
                     space_out=False,
                     vol_factor=1.0):

    if base_center is None:
        base_center = [0, 0, 0]

    layer_df = create_layer_nodes(slices=slices,
                                  rings=rings,
                                  space_out=space_out,
                                  vol_factor=vol_factor)

    liq_df = create_partial_cyl(layer_df=layer_df,
                                layers=liq_layers,
                                cyl_height=liq_level,
                                base_center=base_center,
                                component='Liquid')

    base_center = [base_center[0],
                   base_center[1],
                   base_center[2] + liq_level]

    gas_df = create_partial_cyl(layer_df=layer_df,
                                layers=gas_layers,
                                cyl_height=cyl_height - liq_level,
                                base_center=base_center,
                                component='Gas')

    df = liq_df.append(gas_df, ignore_index=True, sort=False)

    del liq_df, gas_df

    df['radii'] = df['radii'] * cyl_diam / 2

    base_center = [base_center[0],
                   base_center[1],
                   base_center[2] - liq_level]

    df = create_cyl_wall(df)

    df = generate_xyzn(df, base_center=base_center)

    df.reset_index(inplace=True, drop=True)

    df = assign_neighbors(df)

    df = df[['component',
             'theta',
             'radii',
             'height',
             'x',
             'y',
             'z',
             'lft_nbr',
             'rht_nbr',
             'inr_nbr',
             'otr_nbr',
             'delta_theta',
             'delta_radii',
             'delta_height', ]]

    return df


'''
def create_cyl_nodes(slices=1,
                     rings=1,
                     layers=1,
                     cyl_diam=1.0,
                     cyl_height=1.0,
                     base_center=None,
                     space_out=False,
                     vol_factor=1.0,
                     component=None):

    if base_center is None:
        base_center = [0, 0, 0]

    delta_theta = 2 * math.pi / slices
    delta_radii = 1/rings
    delta_height = cyl_height / layers

    lev_nodes = []
    radii = delta_radii / 2
    vol_limit = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta / 2 * vol_factor
    # The volume of the innermost ring * 1.2 is the the maximum volume that an element in a new ring may have.
    # Otherwise, the number of elements in that ring get doubled.

    while radii < 1:
        e_vol = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta/2
        if space_out:
            while e_vol > vol_limit:
                delta_theta /= 2
                e_vol = ((radii + delta_radii / 2) ** 2 - (radii - delta_radii / 2) ** 2) * delta_theta / 2
        theta = delta_theta/2
        while theta < 2 * math.pi:
            lev_nodes.append([theta, radii * cyl_diam / 2, delta_theta])
            theta += delta_theta
        radii += delta_radii

    cyl_nodes = []
    height = delta_height / 2
    while height < cyl_height:
        for lev in range(0, len(lev_nodes)):
            cyl_nodes.append(lev_nodes[lev] + [height])
        height += delta_height

    del lev_nodes

    # TODO test to see if a Pandas-based form of generating the xyz nodes is more efficient.
    xyzn_nodes = []
    for node in cyl_nodes:
        x = base_center[0] + node[1] * math.cos(node[0])
        y = base_center[1] + node[1] * math.sin(node[0])
        z = base_center[2] + node[3]
        n = [x - base_center[0],
             y - base_center[1],
             z - base_center[2]]
        n = np.asarray(n)
        # TODO Replace the normalize vector function with the similar yet more efficient scipy pre-processing function.
        n = normalize_vector(n)
        xyzn_nodes.append([x, y, z, n])

    df = pd.DataFrame(data=np.asarray(cyl_nodes), columns=['theta', 'radii', 'delta_theta', 'height'])

    df['lft_nbr'] = 0
    df['rht_nbr'] = 0
    df['inr_nbr'] = 0
    df['otr_nbr'] = pd.np.empty((len(df), 0)).tolist()

    # Finding the left, right, and inner neighbors.
    for node in range(0, len(df)):
        # finding the left neighbor
        my_df1 = df[df['radii'] == df.loc[node]['radii']]
        my_df2 = my_df1[my_df1['theta'] > df.loc[node]['theta']]
        if len(my_df2) != 0:
            nbr = my_df2[['theta']].idxmin()
        else:
            nbr = my_df1[['theta']].idxmin()
        df.at[node, 'lft_nbr'] = nbr

        # finding the right neighbor
        my_df2 = my_df1[my_df1['theta'] < df.loc[node]['theta']]
        if len(my_df2) != 0:
            nbr = my_df2[['theta']].idxmax()
        else:
            nbr = my_df1[['theta']].idxmax()
        df.at[node, 'rht_nbr'] = nbr

        # finding the inner neighbor
        my_df1 = df[df['radii'] < df.loc[node]['radii']]
        if len(my_df1) != 0:
            my_df1 = my_df1[my_df1['radii'] == my_df1['radii'].max()]
            my_df1['theta_diff'] = abs(my_df1['theta'] - df.loc[node]['theta'])
            nbr = my_df1[['theta_diff']].idxmin()
        else:
            nbr = None
        df.at[node, 'inr_nbr'] = nbr

    df['inr_nbr'] = df['inr_nbr'].astype('Int64')

    # finding the outer neighbors
    for node in range(0, len(df)):
        my_df1 = df[df['inr_nbr'] == node]
        my_df1 = my_df1[my_df1['height'] == df.loc[node]['height']]
        if len(my_df1) != 0:
            nbr = [i for i in list(my_df1.index.values)]
        else:
            nbr = []
        df.at[node, 'otr_nbr'] = nbr

    df_to_merge = pd.DataFrame(np.asarray(xyzn_nodes), columns=['x', 'y', 'z', 'n'])

    df = df.merge(df_to_merge, left_index=True, right_index=True)

    df['delta_radii'] = delta_radii * cyl_diam / 2
    df['delta_height'] = delta_height
    df['otr_area'] = (df['radii'] + df['delta_radii']/2) * df['height']
    df['volume'] = (df['height'] *
                    ((df['radii'] + df['delta_radii']/2)**2 - (df['radii']-df['delta_radii']/2)**2) *
                    df['delta_theta'])
    df['component'] = component

    df = df[['theta',
             'radii',
             'height',
             'x',
             'y',
             'z',
             'n',
             'lft_nbr',
             'rht_nbr',
             'inr_nbr',
             'otr_nbr',
             'delta_theta',
             'delta_radii',
             'delta_height',
             'otr_area',
             'volume']]

    return df
'''


def color_nodes_by_component(component):
    if component == 'Liquid':
        color = 'b'
    elif component == 'Gas':
        color = 'c'
    else:
        color = 'k'
    return color


if __name__ == '__main__':
    """
    
    '''
        ax = plt.gca()
        ax.pie([45, 45, 45, 45, 45, 45, 45, 45], radius=5, wedgeprops={'fc': 'none', 'edgecolor': 'k'})
        for i in range(0, 6):
            circle = plt.Circle((0, 0), radius=i, fill=False, edgecolor='k')
            ax.add_patch(circle)

        angles = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 22.5]

        rings = [0.5, 1.5, 2.5, 3.5, 4.5]

        x = []
        y = []

        for i in angles:
            for j in rings:
                x_coordinate = j * math.cos(math.radians(i))
                y_coordinate = j * math.sin(math.radians(i))
                x.append(x_coordinate)
                y.append(y_coordinate)

        plt.scatter(x, y, color='crimson', marker='.', label='node position')

        #plt.legend(loc='upper center', fancybox=True, facecolor='w')

        plt.axis('scaled')
        plt.show()
    '''  # 2D Nodal Position Plot

    '''

    data = [['L0c0r0', 'L0c1r0', 'L0c2r0', 0],
            ['L0c0r1', 'L0c1r1', 'L0c2r1', 1],
            ['L0c0r2', 'L0c1r2', 'L0c2r2', 2]]
    df1 = pd.DataFrame(data=data, columns=[0, 1, 2, 'row'])
    df1['interval'] = 0

    data = [['L1c0r0', 'L1c1r0', 'L1c2r0', 0],
            ['L1c0r1', 'L1c1r1', 'L1c2r1', 1],
            ['L1c0r2', 'L1c1r2', 'L1c2r2', 2]]
    df2 = pd.DataFrame(data=data, columns=[0, 1, 2, 'row'])
    df2['interval'] = 1

    df = df1.append(df2)

    del df1, df2, data

    df = df.melt(col_level=0, id_vars=['interval', 'row'])
    df.rename(columns={'variable': 'col', 'value': 'Loc'}, inplace=True)
    df.sort_values(['interval', 'col', 'row'], inplace=True)
    df.set_index(['interval', 'col', 'row'], inplace=True)
    print(df)

    df_levs = sorted(list(set(df.index.get_level_values(0))))
    df_cols = sorted(list(set(df.index.get_level_values(1))))
    df_rows = sorted(list(set(df.index.get_level_values(2))))

    df['Temp'] = 0

    for level in df_levs:
        for col in df_cols:
            for row in df_rows:
                print(get_df_value(df_to_search=df,
                                   idx0=level + 1,
                                   idx1=col + 1,
                                   idx2=row + 1,
                                   column='Loc'))

    df.reset_index(inplace=True, drop=True)
    '''  # Temperature Dataframe in hierarchical indexing

    '''
    ans = vf_plane_to_cyl1(s=20, r=10, l=10, t=20, n=100)

    print(ans)

    ans = vf_plane_to_cyl2(s=20, r=10, l=10, t=20, n=100)

    print(ans)
    '''  # Comparison of View Factor Formulas

    '''
    p1 = np.array([1, 2, 3])
    p2 = np.array([4, 6, 9])
    p3 = np.array([12, 11, 9])
    p4 = np.array([0, 0, -15/17])

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    #  print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

    ans = points_all_on_plane(p1=p1, p2=p2, p3=p3, p4=p4)

    print(ans)
    '''  # Testing for vector normalization function and is_plane function.
    
    """  # Older Testing

    pd.set_option('display.max_rows', 2000)
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.width', 2000)

    from PostOffice import *

    ans = create_cyl_nodes(rings=3,
                           slices=4,
                           gas_layers=4,
                           liq_layers=3,
                           cyl_diam=20,
                           cyl_height=3,
                           liq_level=0.5,
                           base_center=[0, 0, 0],
                           space_out=True,
                           vol_factor=1.5)

    try:
        export_results(dfs=[ans], df_names=['Testing'], open_after=True, index=True)
    except PermissionError:
        print('File is locked for editing by user.\nNode network could not be exported.')

    ans['c'] = ans['component'].apply(lambda cpnt: color_nodes_by_component(cpnt))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ans.x.to_list(),
               ans.y.to_list(),
               ans.z.to_list(),
               c=ans.c.to_list(),
               s=5)
    ax.set_axis_off()
    plt.show()
