import math
import pandas as pd
import numpy as np


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
                       vol_factor=1.2,
                       cyl_diam=np.float64(1.0)):

    delta_theta = 2 * math.pi / slices
    delta_radii = 1/rings
    lev_nodes = []
    radii = delta_radii / 2
    vol_limit = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta / 2 * vol_factor
    # The volume of the innermost ring * 1.2 is the the maximum volume that an element in a new ring may have.
    # Otherwise, the number of elements in that ring get doubled.

    while radii < 1:
        e_vol = ((radii+delta_radii/2)**2 - (radii-delta_radii/2)**2) * delta_theta/2
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

    df['radii'] = df['radii'] * cyl_diam / 2

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
        df.at[i, 'n'] = [delta_x, delta_y, 0]
    return df


def create_partial_cyl(layer_df,
                       layers=1,
                       cyl_height=1.0,
                       base_center=None,
                       comp=None):

    delta_height = (cyl_height - base_center[2])/layers
    height = delta_height/2 + base_center[2]
    cyl_df = pd.DataFrame(columns=['theta', 'radii', 'delta_theta', 'height'])
    my_layer_df = pd.DataFrame(columns=['theta', 'radii', 'delta_theta'])

    while height < cyl_height:
        my_layer_df = layer_df.copy()
        my_layer_df['height'] = height
        cyl_df = cyl_df.append(my_layer_df, ignore_index=True, sort=False)
        height += delta_height

    cyl_df['comp'] = comp
    del my_layer_df
    cyl_df.reset_index(inplace=True, drop=True)

    cyl_df['delta_height'] = delta_height

    return cyl_df


def create_cyl_wall(df, wall_thickness=None):
    if df is None:
        print('No data frame to create cylinder wall from.')
        quit()
    if wall_thickness is None:
        wall_thickness = df['delta_radii'].max()
    outer_cyl_nodes = df.loc[df['radii'] == df['radii'].max()].copy()
    outer_cyl_nodes['radii'] = outer_cyl_nodes['radii'] + wall_thickness
    outer_cyl_nodes['delta_radii'] = wall_thickness
    outer_cyl_nodes.comp = 'Wall'
    df = df.append(outer_cyl_nodes, ignore_index=True, sort=False)
    df.reset_index(inplace=True, drop=True)
    return df


def assign_neighbors(df):
    df['lft_nbr'] = 0
    df['rht_nbr'] = 0
    df['inr_nbr'] = 0
    df['otr_nbr_1'] = 0
    df['otr_nbr_2'] = 0

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
        my_df1 = df[
            (df['radii'] < df.loc[node]['radii']) &
            (df['height'] == df.loc[node]['height'])
        ]
        if len(my_df1) != 0:
            my_df1 = my_df1[my_df1['radii'] == my_df1['radii'].max()]
            my_df1 = my_df1[my_df1['z'] == df.loc[node]['z']]
            my_df1['theta_diff'] = abs(my_df1['theta'] - df.loc[node]['theta'])
            nbr = my_df1[['theta_diff']].idxmin()
        else:
            nbr = -1

        df.at[node, 'inr_nbr'] = nbr

    df['inr_nbr'] = df['inr_nbr'].astype('Int64')

    # finding the outer neighbors
    for node in range(0, len(df)):
        my_df1 = df[df['inr_nbr'] == node]
        if len(my_df1) == 1:
            df.at[node, 'otr_nbr_1'] = list(my_df1.index.values)[0]
            df.at[node, 'otr_nbr_2'] = list(my_df1.index.values)[0]
        elif len(my_df1) == 2:
            df.at[node, 'otr_nbr_1'] = list(my_df1.index.values)[0]
            df.at[node, 'otr_nbr_2'] = list(my_df1.index.values)[1]
        elif len(my_df1) == 0:
            df.at[node, 'otr_nbr_1'] = -1
            df.at[node, 'otr_nbr_2'] = -1

    for node in range(0, len(df)):
        if df.loc[node]['otr_nbr_1'] == -1:
            df.at[node, 'otr_nbr_1'] = node
        if df.loc[node]['otr_nbr_2'] == -1:
            df.at[node, 'otr_nbr_2'] = node
        if df.loc[node]['inr_nbr'] == -1:
            df.at[node, 'inr_nbr'] = node

    return df


def create_cyl_nodes(slices=1,
                     rings=1,
                     gas_layers=1,
                     liq_layers=1,
                     cyl_diam=1.0,
                     cyl_height=1.0,
                     liq_level=0.5,
                     base_center=None,
                     vol_factor=1.0,
                     wall_thickness=None,
                     remove_gas_nodes=True):

    if base_center is None:
        base_center = [0, 0, 0]

    if gas_layers == 0:
        cyl_height = liq_level

    layer_df = create_layer_nodes(slices=slices,
                                  rings=rings,
                                  vol_factor=vol_factor,
                                  cyl_diam=cyl_diam)

    liq_df = create_partial_cyl(layer_df=layer_df,
                                layers=liq_layers,
                                cyl_height=liq_level,
                                base_center=base_center,
                                comp='Liquid')

    if gas_layers != 0:
        base_center = [base_center[0],
                       base_center[1],
                       base_center[2] + liq_level]

        gas_df = create_partial_cyl(layer_df=layer_df,
                                    layers=gas_layers,
                                    cyl_height=cyl_height,
                                    base_center=base_center,
                                    comp='Gas')

        df = liq_df.append(gas_df, ignore_index=True, sort=False)
        del liq_df, gas_df

        base_center = [base_center[0],
                       base_center[1],
                       base_center[2] - liq_level]

    else:
        df = liq_df
        del liq_df

    df['lft_theta'] = df['theta'] + df['delta_theta'] / 2
    df['rht_theta'] = df['theta'] - df['delta_theta'] / 2

    if wall_thickness is not None:
        df = create_cyl_wall(df, wall_thickness=wall_thickness)

    #  Outer area is zero if the node is not on the outside of the cylinder
    df['otr_area'] = (df['radii'] + df['delta_radii'] / 2) * df['delta_theta'] * df['delta_height'] * (df['radii'] == df['radii'].max())
    df['volume'] = 4 * df['delta_radii'] * df['radii'] * (df['lft_theta'] - df['rht_theta']) ** 2 ** .5 / 2

    df['otr_radii'] = df['radii'] + df['delta_radii']/2
    df['inr_radii'] = df['radii'] - df['delta_radii']/2

    df = generate_xyzn(df, base_center=base_center)

    if remove_gas_nodes:
        df = df[df['comp'] != 'Gas']

    df.reset_index(inplace=True, drop=True)

    df = assign_neighbors(df)

    df = classify_cylinder_nodes(df)

    df = df[['comp',
             'node_class',
             'theta',
             'radii',
             'height',
             'x',
             'y',
             'z',
             'n',
             'lft_nbr',
             'rht_nbr',
             'inr_nbr',
             'otr_nbr_1',
             'otr_nbr_2',
             'delta_theta',
             'delta_radii',
             'delta_height',
             'inr_radii',
             'otr_radii',
             'lft_theta',
             'rht_theta',
             'otr_area',
             'volume']]

    print('Created {0} nodes.\n'.format(len(df)))

    return df


def assign_node_view_factor(df, cyl_view_factor):
    df['node_vf'] = (np.vectorize(math.cos)(df['rht_theta']) -
                     np.vectorize(math.cos)(df['lft_theta']))
    df['node_vf'] = df['node_vf'] * (df['radii'] == df['radii'].max())
    df['node_vf'] = df['node_vf'] * ((df['theta'] > 0) & (df['theta'] < np.pi))
    df['node_vf'] = df['node_vf'].apply(lambda x: abs(x))
    df['node_vf'] = df['node_vf'] * cyl_view_factor

    return df


def classify_cylinder_nodes(df):

    """

    :param df:
    :return:
    """

    node_classes = {1: 'Liquid Internal',
                    2: 'Gas Internal',
                    3: 'Liquid at Wall Boundary',
                    4: 'Gas at Wall Boundary',
                    5: 'Wall at Liquid Boundary',
                    6: 'Wall at Gas Boundary',
                    7: 'Wall External'}

    max_fluid_radii = df['radii'].where(df['comp'] == 'Liquid').max()
    min_wall_radii = df['radii'].where(df['comp'] == 'Wall').min()
    max_liquid_height = df['height'].where(df['comp'] == 'Liquid').max()

    def classify_node(height, radii):
        # Case 1: Liquid Internal
        if height <= max_liquid_height and radii < max_fluid_radii:
            node_class = 1

        # Case 2: Gas Internal
        elif height > max_liquid_height and radii < max_fluid_radii:
            node_class = 2

        # Case 3: Liquid at Wall
        elif height <= max_liquid_height and radii == max_fluid_radii:
            node_class = 3

        # Case 4: Gas at Wall
        elif height > max_liquid_height and radii == max_fluid_radii:
            node_class = 4

        # Case 5: Wall at Liquid
        elif height <= max_liquid_height and radii == min_wall_radii:
            node_class = 5

        # Case 6: Wall at Gas
        elif height > max_liquid_height and radii == min_wall_radii:
            node_class = 6

        # Case 7: Wall External
        else:
            node_class = 7

        return node_class

    df['node_class'] = df.apply(
        lambda row: classify_node(row['height'],
                                  row['radii']),
        axis=1
    )

    return df


def create_node_fdm_constants(df,
                              densities,
                              specific_heats,
                              thermal_conductivities,
                              delta_time):

    """
    A function to calculate the finite difference constants for each node,
        according to whether they exist on an interface or not.
    :param df: The Pandas dataframe for which the constants for the
        finite difference element constants need to be calculated for.
    :param densities: A dictionary of densities
        (keys: 'Wall', 'Liquid', 'Gas')
    :param specific_heats: A dictionary of specific heat capacities
        (keys: 'Wall', 'Liquid', 'Gas')
    :param thermal_conductivities: A dictionary of thermal conductivities
        (keys: 'Wall', 'Liquid', 'Gas')
    :param delta_time: The real-world time elapsed between one iteration and the next.
    :return: The modified dataframe with the finite difference method constants.
    """

    # Gets rid of the setting with copy warning
    pd.options.mode.chained_assignment = None # default='warn'

    fluid_delta_radii = df.iloc[0]['delta_radii']
    wall_thickness = df[df['comp'] == 'Wall']['delta_radii'].max()

    df['rho'] = df['comp'].map(densities)
    df['Cp'] = df['comp'].map(specific_heats)
    df['k'] = df['comp'].map(thermal_conductivities)

    node_classes = {1: 'Liquid Internal',
                    2: 'Gas Internal',
                    3: 'Liquid at Wall Boundary',
                    4: 'Gas at Wall Boundary',
                    5: 'Wall at Liquid Boundary',
                    6: 'Wall at Gas Boundary'}

    # Calculating the finite difference constants for all the nodes
    df['d1a'] = df['k'] * delta_time / df['rho'] / df['Cp'] / fluid_delta_radii ** 2
    df['d1b'] = df['k'] * delta_time / df['rho'] / df['Cp'] / fluid_delta_radii ** 2
    df['d2'] = df['k'] * delta_time / df['rho'] / df['Cp'] / 2 / df['radii'] / df['delta_radii']

    temp_col = ['ro', 'rb', 'ri', 'ro_sqd', 'rb_sqd', 'ri_sqd', 'Cp_ave_rho_ave_product', 'k_ave']
    df = df.assign(**{i: np.float64(0) for i in temp_col})

    # Start of calculating the finite difference constants for the liquid at wall nodes
    if True:
        sub_df = (df['node_class'] == 3)

        # Start of d1 calculations
        if True:
            df['ro'].loc[sub_df] = wall_thickness / 2 + df['otr_radii'].loc[sub_df]
            df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

            df['rb'].loc[sub_df] = df['otr_radii'].loc[sub_df]
            df['rb_sqd'].loc[sub_df] = df['rb'].loc[sub_df] * df['rb'].loc[sub_df]

            df['ri'].loc[sub_df] = df['inr_radii'].loc[sub_df]
            df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

            df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                densities['Wall'] * specific_heats['Wall'] *
                (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                densities['Liquid'] * specific_heats['Liquid'] *
                (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
            ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

            df['k_ave'].loc[sub_df] = \
                np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Liquid'] +
                 np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall'])

            df['d1a'].loc[sub_df] = \
                df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * delta_time / \
                df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

            df['d1b'].loc[sub_df] = df['k'].loc[sub_df] * delta_time / df['rho'].loc[sub_df] / \
                df['Cp'].loc[sub_df] / fluid_delta_radii ** 2

        # Start of d2 calculations
        if True:
            df['ri'].loc[sub_df] = df['inr_radii'].loc[sub_df] - fluid_delta_radii
            df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

            df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                densities['Wall'] * specific_heats['Wall'] *
                (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                densities['Liquid'] * specific_heats['Liquid'] *
                (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
            ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

            df['k_ave'].loc[sub_df] = \
                np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Liquid'] +
                 np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall'])

            df['d2'].loc[sub_df] = \
                df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * \
                delta_time / 2 / df['delta_radii'].loc[sub_df] / df['radii'].loc[sub_df]

    # Start of calculating the finite difference constants for the gas at wall nodes
    if True:
        sub_df = (df['node_class'] == 4)

        # Start of d1 calculations
        if True:
            df['ri'].loc[sub_df] = df['inr_radii'].loc[sub_df]
            df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

            df['ro'].loc[sub_df] = df['otr_radii'].loc[sub_df]
            df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

            df['rb'].loc[sub_df] = df['otr_radii'].loc[sub_df] + fluid_delta_radii / 2
            df['rb_sqd'].loc[sub_df] = df['rb'].loc[sub_df] * df['rb'].loc[sub_df]

            df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                densities['Wall'] * specific_heats['Wall'] *
                (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                densities['Gas'] * specific_heats['Gas'] *
                (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
            ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

            df['k_ave'].loc[sub_df] = \
                np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) /
                 thermal_conductivities['Gas'] +
                 np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) /
                 thermal_conductivities['Wall'])

            df['d1a'].loc[sub_df] = df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] \
                * delta_time / df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

            df['d1b'].loc[sub_df] = df['k'].loc[sub_df] * delta_time / df['rho'].loc[sub_df] / \
                df['Cp'].loc[sub_df] / fluid_delta_radii ** 2

        # Start of d2 calculations
        if True:
            df['ri'].loc[sub_df] = df['inr_radii'] - fluid_delta_radii
            df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

            df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                densities['Wall'] * specific_heats['Wall'] *
                (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                densities['Gas'] * specific_heats['Gas'] *
                (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
            ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

            df['k_ave'].loc[sub_df] = \
                np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) /
                 thermal_conductivities['Gas'] +
                 np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) /
                 thermal_conductivities['Wall'])

            df['d2'].loc[sub_df] = \
                df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * \
                delta_time / 2 / df['delta_radii'].loc[sub_df] / df['radii'].loc[sub_df]

        # Start of calculating the finite difference constants for the wall at liquid nodes
        if True:
            sub_df = (df['node_class'] == 5)
            # Start of d1 calculations
            if True:
                df['d1a'].loc[sub_df] = df['k'].loc[sub_df] * delta_time / \
                                        df['rho'].loc[sub_df] / df['Cp'].loc[sub_df] / fluid_delta_radii ** 2

                df['ro'].loc[sub_df] = df['radii'].loc[sub_df]
                df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

                df['rb'].loc[sub_df] = df['inr_radii'].loc[sub_df]
                df['rb_sqd'].loc[sub_df] = df['rb'].loc[sub_df] * df['rb'].loc[sub_df]

                df['ri'].loc[sub_df] = df['inr_radii'].loc[sub_df] - fluid_delta_radii / 2
                df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

                df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                    densities['Wall'] * specific_heats['Wall'] *
                    (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                    densities['Liquid'] * specific_heats['Liquid'] *
                    (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])) / \
                    (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

                df['k_ave'].loc[sub_df] = \
                    np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / (
                            np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Liquid'] +
                            np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall']
                    )

                df['d1b'].loc[sub_df] = df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * \
                    delta_time / df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

            # Start of d2 calculations
            if True:
                df['ro'].loc[sub_df] = df['radii'].loc[sub_df] + wall_thickness
                df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

                df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                    densities['Wall'] * specific_heats['Wall'] *
                    (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                    densities['Liquid'] * specific_heats['Liquid'] *
                    (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
                ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

                df['k_ave'].loc[sub_df] = \
                    np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                    (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Liquid'] +
                     np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall'])

                df['d2'].loc[sub_df] = \
                    df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * delta_time / \
                    df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

        # Start of calculating the finite difference constants for the wall at gas nodes
        if True:
            sub_df = (df['node_class'] == 6)

            # Start of d1 calculations
            if True:
                df['d1a'].loc[sub_df] = df['k'].loc[sub_df] * delta_time / df['rho'].loc[sub_df] / \
                                        df['Cp'].loc[sub_df] / fluid_delta_radii ** 2

                df['ro'].loc[sub_df] = df['radii'].loc[sub_df]
                df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

                df['rb'].loc[sub_df] = df['inr_radii'].loc[sub_df]
                df['rb_sqd'].loc[sub_df] = df['rb'].loc[sub_df] * df['rb'].loc[sub_df]

                df['ri'].loc[sub_df] = df['inr_radii'].loc[sub_df] - fluid_delta_radii / 2
                df['ri_sqd'].loc[sub_df] = df['ri'].loc[sub_df] * df['ri'].loc[sub_df]

                df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                    densities['Wall'] * specific_heats['Wall'] *
                    (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                    densities['Gas'] * specific_heats['Gas'] *
                    (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
                ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

                df['k_ave'].loc[sub_df] = \
                    np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / \
                    (np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Gas'] +
                     np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall'])

                df['d1b'].loc[sub_df] = df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * \
                    delta_time / df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

            # Start of d2 calculations
            if True:
                df['ro'].loc[sub_df] = df['radii'].loc[sub_df] + wall_thickness
                df['ro_sqd'].loc[sub_df] = df['ro'].loc[sub_df] * df['ro'].loc[sub_df]

                df['Cp_ave_rho_ave_product'].loc[sub_df] = (
                    densities['Wall'] * specific_heats['Wall'] *
                    (df['ro_sqd'].loc[sub_df] - df['rb_sqd'].loc[sub_df]) +
                    densities['Gas'] * specific_heats['Gas'] *
                    (df['rb_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])
                ) / (df['ro_sqd'].loc[sub_df] - df['ri_sqd'].loc[sub_df])

                df['k_ave'].loc[sub_df] = \
                    np.log(df['ro'].loc[sub_df] / df['ri'].loc[sub_df]) / (
                    np.log(df['rb'].loc[sub_df] / df['ri'].loc[sub_df]) / thermal_conductivities['Gas'] +
                    np.log(df['ro'].loc[sub_df] / df['rb'].loc[sub_df]) / thermal_conductivities['Wall']
                    )

                df['d2'].loc[sub_df] = df['k_ave'].loc[sub_df] / df['Cp_ave_rho_ave_product'].loc[sub_df] * \
                    delta_time / df['delta_radii'].loc[sub_df] / df['delta_radii'].loc[sub_df]

        df['d3'] = df['k'] * delta_time / df['radii'] / df['radii'] / df['delta_theta'] / df['delta_theta']

    pd.options.mode.chained_assignment = 'warn'

    return df

