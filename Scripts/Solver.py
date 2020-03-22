import sys
import traceback
from SetUp import *
from Graphics import *
from PostOffice import *
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


def update_node_temp_ft(func_df, delta_time, tick, tock, h_values, local_temps, str_times):
    tick_T_col = 'T @ ' + str_times[tick]
    tock_T_col = 'T @ ' + str_times[tock]
    tick_HFe_col = 'External Heat Flux @ ' + str_times[tick]
    tick_HFi_col = 'Internal Heat Flux @ ' + str_times[tick]
    tick_HG_col = 'Heat Gen @ ' + str_times[tick]

    func_df[tick_HFe_col] = (
        (5.67e-8 * (local_temps['fire_temp'] ** 4 - func_df[tick_T_col] ** 4) * func_df['node_vf'] +
         (local_temps['amb_temp'] - func_df[tick_T_col]) * h_values['tank_exterior'])
        ) * delta_time

    func_df[tick_HFi_col] = (local_temps['amb_temp'] - func_df[tick_T_col]) * h_values['tank_interior'] * delta_time

    func_df[tick_HG_col] = (func_df[tick_HFe_col] * func_df['otr_area'] + func_df[tick_HFi_col] * func_df['inr_area']) / func_df['volume']

    T_otr1 = func_df.loc[func_df['otr_nbr_1'], tick_T_col].to_numpy()
    T_otr2 = func_df.loc[func_df['otr_nbr_2'], tick_T_col].to_numpy()
    T_inr = func_df.loc[func_df['inr_nbr'], tick_T_col].to_numpy()
    T_lft = func_df.loc[func_df['lft_nbr'], tick_T_col].to_numpy()
    T_rht = func_df.loc[func_df['rht_nbr'], tick_T_col].to_numpy()
    T_otr = (T_otr1 + T_otr2)/2

    func_df[tock_T_col] = \
        func_df['d1a'] * (T_otr - func_df[tick_T_col]) + \
        func_df['d1b'] * (T_inr - func_df[tick_T_col]) + \
        func_df['d2'] * (T_otr - T_inr) + \
        func_df['d3'] * (T_rht - 2 * func_df[tick_T_col] + T_lft) + \
        func_df[tick_HG_col] / func_df['rho'] / func_df['Cp'] + \
        func_df[tick_T_col]

    return func_df


def ss_error(func_df):

    T_otr1 = func_df.loc[func_df['otr_nbr_1'], 'Temp'].to_numpy()
    T_otr2 = func_df.loc[func_df['otr_nbr_2'], 'Temp'].to_numpy()
    T_inr = func_df.loc[func_df['inr_nbr'], 'Temp'].to_numpy()
    T_lft = func_df.loc[func_df['lft_nbr'], 'Temp'].to_numpy()
    T_rht = func_df.loc[func_df['rht_nbr'], 'Temp'].to_numpy()
    T_otr = (T_otr1 + T_otr2) / 2


    func_df['B'] = func_df['d1a'].to_numpy() + \
                   func_df['d1b'].to_numpy() + \
                   2 * func_df['d3'].to_numpy()

    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['C'] = func_df['d1a'].to_numpy() * T_otr + \
                   func_df['d1b'].to_numpy() * T_inr + \
                   func_df['d2'].to_numpy() * (T_otr - T_inr) + \
                   func_df['d3'].to_numpy() * (T_rht + T_lft)

    error = func_df['C'].to_numpy()/func_df['B'].to_numpy() - func_df['Temp'].to_numpy()

    return error


def preprocess_node_temp_ss(func_df, h_values):
    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['A'] = func_df['node_vf'].to_numpy() * 5.67e-8 * func_df['otr_area'].to_numpy() / func_df['volume'].to_numpy()

    # Since fund_df['otr_area'] should be zero for all non-wall nodes, this assignment should work for all nodes.
    func_df['B'] = func_df['d1a'].to_numpy() + \
                   func_df['d1b'].to_numpy() + \
                   2 * func_df['d3'].to_numpy() + \
                   h_values['tank_exterior'] * func_df['otr_area'].to_numpy() / func_df['volume'].to_numpy()

    return func_df


def update_node_temp_ss(func_df, h_values, local_temps):

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
                   (func_df['otr_area'].to_numpy() * (
                           func_df['node_vf'].to_numpy() * 5.67e-8 * local_temps['fire_temp'] ** 4 +
                           h_values['tank_exterior'] * local_temps['amb_temp']
                   ) + func_df['inr_area'].to_numpy() * h_values['tank_interior'] * local_temps['amb_temp']
                    ) / func_df['volume'].to_numpy()

    temp = (root4improved(func_df['A'].to_numpy(dtype=complex),
                          func_df['B'].to_numpy(dtype=complex),
                          func_df['C'].to_numpy(dtype=complex))
                       ).round(5)

    return temp



# Old Steady State Function
'''
def update_node_temp_ss(func_df, h_values, local_temps):

    func_df['Heat Gen'] = \
        ((5.67e-8 * (
                local_temps['fire_temp'] ** 4 - func_df['Temp'] ** 4
        ) * func_df['node_vf'] +
         (local_temps['amb_temp'] - func_df['Temp']) * h_values['tank_exterior'])
        ) * func_df['otr_area'] / func_df['volume']

    T_otr1 = func_df.loc[func_df['otr_nbr_1'], 'Temp'].to_numpy()
    T_otr2 = func_df.loc[func_df['otr_nbr_2'], 'Temp'].to_numpy()
    T_inr = func_df.loc[func_df['inr_nbr'], 'Temp'].to_numpy()
    T_lft = func_df.loc[func_df['lft_nbr'], 'Temp'].to_numpy()
    T_rht = func_df.loc[func_df['rht_nbr'], 'Temp'].to_numpy()
    T_otr = (T_otr1 + T_otr2)/2

    func_df['Temp'] = (
        func_df['d1a'] * T_otr +
        func_df['d1b'] * T_inr +
        func_df['d2'] * (T_otr + T_inr) +
        func_df['d3'] * (T_rht + T_lft) +
        func_df['Heat Gen'] / func_df['rho'] / func_df['Cp']
        )/(
        func_df['d1a'].to_numpy() + func_df['d1b'].to_numpy() + func_df['d3'].to_numpy() + func_df['d3'].to_numpy()
        )

    return func_df
    '''


if __name__ == '__main__':
    try:
        if True:
            pd.set_option('display.max_rows', 2000)
            pd.set_option('display.max_columns', 2000)
            # pd.set_option('display.width', 2000)

            inputs = import_cases_and_fluids()

            mf_list = {
                k: inputs[k] for k in inputs.keys() &
                {'C3', 'IC4', 'NC4', 'IC5', 'NC5', 'C6', 'C7+'}
            }

            liq_k, liq_Cp, liq_rho = mix_me(mf_list)

            comp_Cps = {'Liquid': liq_Cp,
                        'Gas': 1.169627,
                        'Wall': inputs['Wall_Cp[J/kgK]']}

            comp_ks = {'Liquid': liq_k,
                       'Gas': 0.0543496649,
                       'Wall': inputs['Wall_k[W/mK]']}

            comp_rhos = {'Liquid': liq_rho,
                         'Gas': 0.1,
                         'Wall': inputs['Wall_rho[kg/m3]']}

            print('Importing and pre-processing...\n')

            tank_od = inputs['TankID[m]'] + inputs['WallThickness[cm]'] / 100

            vf = vf_plane_to_cyl2(s=inputs['FireDistanceFromTankCenter[m]'],
                                  r=tank_od / 2,
                                  l=inputs['TankHeight[m]'],
                                  t=tank_od,
                                  n=100)

            print('Creating cylinder...\n')

            if inputs['FluidLevel[m]'] >= inputs['TankHeight[m]']:
                gas_layers = 0
            else:
                gas_layers = 1

            node_df = create_cyl_nodes(rings=inputs['Rings'],
                                       slices=inputs['Slices'],
                                       gas_layers=gas_layers,
                                       liq_layers=1,
                                       cyl_diam=tank_od,
                                       cyl_height=inputs['TankHeight[m]'],
                                       liq_level=inputs['FluidLevel[m]'],
                                       base_center=[0, 0, 0],
                                       vol_factor=inputs['vol_factor'],
                                       wall_thickness=inputs['WallThickness[cm]'] / 100,
                                       remove_gas_nodes=True)

            '''
            if True:
                print('Building node visual...')
                node_df['c'] = node_df['comp'].apply(lambda cpnt: color_nodes_by_component(cpnt))
                generate_3d_node_geometry(prop_df=node_df)
            '''

            node_df = assign_node_view_factor(
                df=node_df,
                cyl_view_factor=vf,
                cyl_emissivity=inputs['Emissivity']
            )

            if inputs['Mode'] == 'Steady_State':
                node_df = create_node_fdm_constants(
                    df=node_df,
                    densities=comp_rhos,
                    specific_heats=comp_Cps,
                    thermal_conductivities=comp_ks,
                    delta_time=1
                )
            else:
                node_df = create_node_fdm_constants(
                    df=node_df,
                    densities=comp_rhos,
                    specific_heats=comp_Cps,
                    thermal_conductivities=comp_ks,
                    delta_time=inputs['TimeStep[s]']
                )

            nodes = list(node_df.index)
            otr_node_radii = node_df['radii'].max()

            loc_temps = {}
            loc_temps.update({'fire_temp': np.float64(inputs['FireTemp[C]'] + 273.15)})
            loc_temps.update({'amb_temp': np.float64(inputs['Ambient/InitialTemp[C]'] + 273.15)})

            h_vals = {'tank_exterior': inputs['External_Tank_h_value[W/m²K]'],
                      'tank_interior': inputs['Internal_Tank_h_value[W/m²K]']}

        #  generate_3d_node_geometry(df)
        export = False

        if inputs['Mode'] == 'Steady_State':
            print('Starting steady state iterations...')

            t = 0
            not_at_ss = True
            node_df = node_df.assign(**{'Temp': loc_temps['amb_temp']})
            node_df['Error'] = ss_error(node_df)

            #results_df = pd.DataFrame(df[['radii', 'theta', 'volume', 'otr_area', 'Error', 'Temp']])

            #slope_df = pd.DataFrame()
            old_temp = node_df['Temp'].copy(deep=True)

            node_df = preprocess_node_temp_ss(node_df, h_vals)

            while not_at_ss: # and t < 10000:
                old_temp = node_df['Temp'].copy(deep=True)

                node_df['Temp'] = update_node_temp_ss(
                    func_df=node_df,
                    h_values=h_vals,
                    local_temps=loc_temps,
                )

                nodes_at_ss = node_df['Temp'].eq(old_temp).sum()

                print(
                    '\rCurrently on iteration {0}, there are {1}/{2} nodes at steady-state.'.format(
                        int(t), nodes_at_ss, len(node_df)
                    ), end='', flush=True)

                not_at_ss = nodes_at_ss != len(node_df)

                t += 1

            print('\n', not_at_ss, 'at', t, 'iterations.')

        elif inputs['Mode'] == 'Fixed_Time':
            print('Starting fixed time iterations...')

            time_steps = list(range(0, inputs['TimeIterations[#]'] + 1))  #
            str_time_steps = ["t={:0.2f}s".format(
                i * inputs['TimeStep[s]']) for i in time_steps]

            node_df = node_df.assign(**{'T @ ' + str_time_steps[0]: loc_temps['amb_temp']})

            total_time_steps = int(len(time_steps)) - 1

            for t in time_steps[:-1]:
                print('\rCurrently on timestep {0} of {1}.'.format(
                    int(t) + 1, total_time_steps),
                      end='', flush=True)
                node_df = update_node_temp_ft(
                    func_df=node_df,
                    delta_time=inputs['TimeStep[s]'],
                    tick=t,
                    tock=t+1,
                    h_values=h_vals,
                    local_temps=loc_temps,
                    str_times=str_time_steps
                )

        print('\nFinished iterations.\n')

        if export:
            print('Exporting results...\n')
            try:
                if str(inputs['Case_name']) != 'nan':
                    export_results(dfs=[node_df],
                                   df_names=[inputs['Case_name']],
                                   open_after=False,
                                   index=True)

            except PermissionError:
                print('File is locked for editing by user.\n\tNode network could not be exported.')

    except BaseException:
        print(sys.exc_info()[0])
        print(traceback.format_exc())

    finally:
        print("Press Enter to continue ...")
        # input()
