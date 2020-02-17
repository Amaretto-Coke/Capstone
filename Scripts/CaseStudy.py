import pandas as pd
import SetUp as su
import numpy as np

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)


def classify_cylinder_nodes(df):

    node_classes = {1: 'Inner',
                    2: 'Middle',
                    3: 'Outer'}

    max_radii = df['radii'].max()
    min_radii = df['radii'].min()

    def classify_node(radii):
        # Case 1: Liquid Internal
        if radii == max_radii:
            node_class = 3
        elif radii == min_radii:
            node_class = 1
        else:
            node_class = 2
        return node_class

    df['node_class'] = df.apply(lambda row: classify_node(row['radii']), axis=1)

    return df


#  Unfinished:
def create_node_fdm_constants(df,
                              densities,
                              specific_heats,
                              thermal_conductivities,
                              delta_time):

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


if __name__ == '__main__':
    df = su.create_layer_nodes(slices=6, rings=3)

    df.pop('delta_radii')

    df['height'] = np.ones(len(df))

    df = su.generate_xyzn(df)

    df = su.assign_neighbors(df)

    original_radii = set(df['radii'].to_list())

    #  Literature example had the condition of r1/r0=2, r2/r0=4, and r3/r0=6.
    #  If we make r0 = 1, then [r1, r3] equals [2, 6].

    new_radii = [1, 2, 6]

    radii_dict = dict(zip(original_radii, new_radii))

    df['radii'] = df['radii'].replace(radii_dict)




