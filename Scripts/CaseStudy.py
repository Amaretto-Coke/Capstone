import os
import math
import SetUp as su
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)


def k_ave(k_inr, k_otr, ri, rb, ro):
    return np.log(ro/ri)/(np.log(rb/ri)/k_inr + np.log(ro/rb)/k_otr)


def rho_ave(rho_inr, rho_otr, ri, rb, ro):
    return (rho_otr * (ro**2 - rb**2) + rho_inr * (rb**2 - ri**2))/(ro**2 - ri**2)


def d1a(d1a_df):
    d1a_rb = d1a_df['otr_radii'].to_numpy()
    d1a_ri = d1a_df['radii'].to_numpy()
    d1a_ro = d1a_df['d1a_ro'].to_numpy()

    d1a_rho_otr = d1a_df['otr_rho'].to_numpy()
    d1a_rho_inr = d1a_df['inr_rho'].to_numpy()

    d1a_k_inr = d1a_df['inr_k'].to_numpy()
    d1a_k_otr = d1a_df['otr_k'].to_numpy()

    d1a_k_ave = k_ave(k_inr=d1a_k_inr, k_otr=d1a_k_otr, ri=d1a_ri, rb=d1a_rb, ro=d1a_ro)
    d1a_rho_ave = rho_ave(rho_inr=d1a_rho_inr, rho_otr=d1a_rho_otr, ri=d1a_ri, rb=d1a_rb, ro=d1a_ro)

    delta_time = 1
    Cp = 1

    return d1a_k_ave * delta_time / d1a_rho_ave / Cp / (d1a_ro-d1a_ri) ** 2


def d1b(d1b_df):
    d1b_rb = d1b_df['inr_radii'].to_numpy()
    d1b_ri = d1b_df['d1b_ri'].to_numpy()
    d1b_ro = d1b_df['radii'].to_numpy()

    d1b_rho_otr = d1b_df['otr_rho'].to_numpy()
    d1b_rho_inr = d1b_df['inr_rho'].to_numpy()

    d1b_k_inr = d1b_df['inr_k'].to_numpy()
    d1b_k_otr = d1b_df['otr_k'].to_numpy()

    d1b_k_ave = k_ave(k_inr=d1b_k_inr, k_otr=d1b_k_otr, ri=d1b_ri, rb=d1b_rb, ro=d1b_ro)
    d1b_rho_ave = rho_ave(rho_inr=d1b_rho_inr, rho_otr=d1b_rho_otr, ri=d1b_ri, rb=d1b_rb, ro=d1b_ro)

    delta_time = 1
    Cp = 1

    return d1b_k_ave * delta_time / d1b_rho_ave / Cp / (d1b_ro-d1b_ri) ** 2


def d2(d2_df):

    return 4


def preprocess_node_temp_ss(func_df):
    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['A'] = func_df['node_vf'].to_numpy() * 5.67e-8 * func_df['otr_area'].to_numpy() / func_df['volume'].to_numpy()

    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['B'] = func_df['d1a'].to_numpy() + \
                   func_df['d1b'].to_numpy() + \
                   2 * func_df['d3'].to_numpy()

    return func_df


def update_node_temp_ss(func_df):

    def root4improved(a, b, c):
        A = (3**0.5*(256*a**3*c**3+27*a**2*b**4)**0.5+9*a*b**2)**(1/3)
        B = (A/(2**(1/3)*3**(2/3)*a)-(4*(2/3)**(1/3)*c)/A)**0.5
        C = 1/2*((2*b)/(a*B)-A/(2**(1/3)*3**(2/3)*a)+(4*(2/3)**(1/3)*c)/A)**0.5-1/2*B
        return np.where(a != 0, np.real(C), np.real(c/b))

    T_otr1 = func_df.loc[func_df['otr_nbr_1'], 'Temp'].to_numpy()
    T_otr2 = func_df.loc[func_df['otr_nbr_2'], 'Temp'].to_numpy()
    T_inr = func_df.loc[func_df['inr_nbr'], 'Temp'].to_numpy()
    T_lft = func_df.loc[func_df['lft_nbr'], 'Temp'].to_numpy()
    T_rht = func_df.loc[func_df['rht_nbr'], 'Temp'].to_numpy()
    T_otr = (T_otr1 + T_otr2) / 2

    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['C'] = func_df['d1a'].to_numpy() * T_otr + \
                   func_df['d1b'].to_numpy() * T_inr + \
                   func_df['d2'].to_numpy() * (T_otr - T_inr) + \
                   func_df['d3'].to_numpy() * (T_rht + T_lft) + \
                   func_df['otr_area'].to_numpy() / func_df['volume'].to_numpy() * (
                           func_df['node_vf'].to_numpy()
                   )

    temp = (root4improved(func_df['A'].to_numpy(dtype=complex),
                          func_df['B'].to_numpy(dtype=complex),
                          func_df['C'].to_numpy(dtype=complex))
                       ).round(5)

    return temp


def assign_node_view_factor(df):

    def integral_up_to_theta(theta):
        return np.pi**2 * theta**3 / 3 - (.5 * np.pi * theta**4) + (theta**5)/5

    df['lft_theta'] = df['theta'] + df['delta_theta']/2
    df['rht_theta'] = df['theta'] - df['delta_theta']/2

    df['node_vf'] = (
            integral_up_to_theta(df['lft_theta'].to_numpy()) -
            integral_up_to_theta(df['rht_theta'].to_numpy())
    )/df['delta_theta'].to_numpy()

    df['node_vf'] = df['node_vf'] * (df['radii'] == df['radii'].max())
    df['node_vf'] = df['node_vf'] * ((df['theta'] > 0) & (df['theta'] < np.pi))
    df['node_vf'] = df['node_vf'].apply(lambda x: abs(x))
    df['node_vf'] = df['node_vf'] * 3

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


def animate(i):
    pullData = open('tmp_output.txt', 'r').read()


if __name__ == '__main__':
    # Creating a path string to the user interface excel file.
    path = os.path.dirname(os.getcwd()) + r'\MailBox.xlsx'

    # Importing the two Excel sheets as two new data frames.
    inputs = pd.read_excel(path, sheet_name='CaseStudy')

    df = su.create_layer_nodes(slices=6, rings=4)

    df.pop('delta_radii')

    df['height'] = np.ones(len(df))

    df = su.generate_xyzn(df)

    df = su.assign_neighbors(df)

    original_radii = set(df['radii'].to_list())
    new_radii = [0.5, 1.5, 3, 5]

    radii_dict = dict(zip(original_radii, new_radii))
    df['radii'] = df['radii'].replace(radii_dict)

    df = pd.merge(df, inputs, how='left', on='radii')

    delta_time = 1

    df['d1a'] = d1a(df)
    df['d1b'] = d1b(df)
    df['d2'] = df['k'] * delta_time / df['rho'] / df['Cp'] / 2 / df['radii'] / df['delta_radii']
    df['d3'] = df['k'] * delta_time / df['rho'] / df['Cp'] / df['radii'] / df['radii'] / df['delta_theta'] / df['delta_theta']

    df = assign_node_view_factor(df)

    t = 0
    not_at_ss = True

    # Sets the initial Temp to be 0
    df = df.assign(**{'Temp': 0})

    old_temp = df['Temp'].copy(deep=True)

    df['otr_area'] = (df['radii'] + df['delta_radii'] / 2) * df['delta_theta'] * (
                df['radii'] == df['radii'].max())
    df['volume'] = 4 * df['delta_radii'] * df['radii'] * (df['lft_theta'] - df['rht_theta']) ** 2 ** .5 / 2

    df = preprocess_node_temp_ss(df)

    while not_at_ss:  # and t < 10000:
        old_temp = df['Temp'].copy(deep=True)

        # Updates the the node temperatures
        df['Temp'] = update_node_temp_ss(df)

        # Sets the internal ring of nodes to have a temperature of zero.
        df['Temp'] = df['Temp'] * (df['radii'] != df['radii'].min())

        nodes_at_ss = df['Temp'].eq(old_temp).sum()

        os.path.dirname(os.getcwd()) + r'\output_df.csv'

        df.to_csv(os.path.dirname(os.getcwd()) + r'\output_df.csv')

        print(
            '\rCurrently on iteration {0}, there are {1}/{2} nodes at steady-state.'.format(
                int(t), nodes_at_ss, len(df)
            ), end='', flush=True)

        not_at_ss = nodes_at_ss != len(df)

        t += 1























