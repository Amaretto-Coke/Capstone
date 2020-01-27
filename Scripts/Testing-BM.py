import sys
import traceback
import time
from SetUp import *
from Graphics import *
from PostOffice import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


def update_node_temp_pd(func_df, delta_time, tick, tock, h_values, local_temps, str_times):
    tick_T_col = 'T @ ' + str_times[tick]
    tock_T_col = 'T @ ' + str_times[tock]
    tick_HF_col = 'Heat Flux @ ' + str_times[tick]
    tick_HG_col = 'Heat Gen @ ' + str_times[tick]

    func_df[tick_HF_col] = \
        ((func_df['node_vf'] * 5.67e-8 * (local_temps['fire_temp'] ** 4 - func_df[tick_T_col] ** 4) +
         (local_temps['amb_temp'] - func_df[tick_T_col]) * h_values['tank_exterior'])) * delta_time

    func_df[tick_HG_col] = func_df[tick_HF_col] * func_df['otr_area'] / func_df['volume']

    T_otr = (func_df.loc[func_df['otr_nbr_1'], tick_T_col] + func_df.loc[func_df['otr_nbr_2'], tick_T_col])/2
    T_inr = func_df.loc[func_df['inr_nbr'], tick_T_col]
    T_lft = func_df.loc[func_df['lft_nbr'], tick_T_col]
    T_rht = func_df.loc[func_df['lft_nbr'], tick_T_col]

    nbr_Ts = [T_otr, T_inr, T_lft, T_rht]

    for T in nbr_Ts:
        T.reset_index(inplace=True, drop=True)

    func_df[tock_T_col] = \
        func_df['d1a'] * (T_otr - func_df[tick_T_col]) + \
        func_df['d1b'] * (T_inr - func_df[tick_T_col]) + \
        func_df['d2'] * (T_otr - T_inr) + \
        func_df['d3'] * (T_rht - 2 * func_df[tick_T_col] + T_lft) + \
        func_df[tick_HG_col] / func_df['rho'] / func_df['Cp'] + \
        func_df[tick_T_col]

    return func_df

if __name__ == '__main__':
    try:
        if True:
            # pd.set_option('display.max_rows', 2000)
            pd.set_option('display.max_columns', 2000)
            # pd.set_option('display.width', 2000)

            inputs = import_cases_and_fluids()
            export = True

            comp_Cps = {'Liquid': inputs['Liquid_Cp[J/gK]'],
                        'Gas': inputs['Air_Cp[J/gK]'],
                        'Wall': inputs['Wall_Cp[J/gK]']}

            comp_ks = {'Liquid': inputs['Liquid_k[W/mK]'],
                       'Gas': inputs['Air_k[W/mK]'],
                       'Wall': inputs['Wall_k[W/mK]']}

            comp_rhos = {'Liquid': inputs['Liquid_rho[kg/m3]'],
                         'Gas': inputs['Air_rho[kg/m3]'],
                         'Wall': inputs['Wall_rho[kg/m3]']}

            print('Importing and pre-processing...\n')

            tank_od = inputs['TankID[m]'] + inputs['WallThickness[cm]'] / 100

            vf = vf_plane_to_cyl2(s=inputs['FireDistanceFromTankCenter[m]'],
                                  r=tank_od / 2,
                                  l=inputs['TankHeight[m]'],
                                  t=tank_od,
                                  n=100)

            print('Creating cylinder...\n')

            node_df = create_cyl_nodes(rings=inputs['rings'],
                                       slices=inputs['slices'],
                                       gas_layers=inputs['gas_layers'],
                                       liq_layers=inputs['liq_layers'],
                                       cyl_diam=tank_od,
                                       cyl_height=inputs['TankHeight[m]'],
                                       liq_level=inputs['FluidLevel[m]'],
                                       base_center=[0, 0, 0],
                                       space_out=inputs['space_out'],
                                       vol_factor=inputs['vol_factor'],
                                       wall_thickness=inputs['WallThickness[cm]'] / 100)

            node_df['c'] = node_df['comp'].apply(lambda cpnt: color_nodes_by_component(cpnt))

            # if inputs['show_geo']:
            if False:
                print('Building node visual...\n')
                generate_3d_node_geometry(prop_df=node_df)

            node_df = assign_node_view_factor(df=node_df, cyl_view_factor=vf)

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

            h_vals = {'tank_exterior': 15, 'tank_interior': 15}

            time_steps = list(range(0, inputs['TimeIterations[#]'] + 1))
            str_time_steps = ["t={:0.2f}s".format(i * inputs['TimeStep[s]']) for i in time_steps]

            # Creates numerous columns in the dataframe, one for every time iteration
            node_df = node_df.assign(**{'T @ ' + str_time_steps[0]: loc_temps['amb_temp']})


            print('Starting time iterations...')

            total_time_steps = int(len(time_steps)) - 1

        for t in time_steps[:-1]:
            print('\rCurrently on timestep {0} of {1}.'.format(
                int(t) + 1, total_time_steps),
                  end='', flush=True)
            node_df = update_node_temp_pd(
                func_df=node_df,
                delta_time=inputs['TimeStep[s]'],
                tick=t,
                tock=t+1,
                h_values=h_vals,
                local_temps=loc_temps,
                str_times=str_time_steps
            )

        print('\nFinished iterations.\n')

        # Call to broken graphics functions
        '''
        print('Making graphics.\n')

        generate_time_gif(temp_df=node_df, prop_df=node_df, time_steps=time_steps)

        generate_boundary_graphs(temp_df=node_df,
                                 prop_df=node_df,
                                 time_steps=time_steps,
                                 features=['heat_flux',
                                           'tock_temp',
                                           'heat_gen'],
                                 labels=['Heat Flux\n[W/m²]',
                                         'Node\nTemperature\n[K]',
                                         'Heat\nGeneration\n[W/m³]'],
                                 color_map='hsv')

        print('\nFinished making graphics.\n')
        '''

        if export:
            print('Exporting results...\n')
            try:
                export_results(dfs=[node_df],
                               df_names=['node_df'],
                               open_after=True,
                               index=True)
            except PermissionError:
                print('File is locked for editing by user.\n\tNode network could not be exported.')

    except BaseException:
        print(sys.exc_info()[0])

        print(traceback.format_exc())
    finally:
        print("Press Enter to continue ...")
        input()


