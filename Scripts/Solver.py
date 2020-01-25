import sys
import traceback
from SetUp import *
from Graphics import *
from PostOffice import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


def update_node_temp(prop_df,
                     temp_df,
                     delta_time,
                     tick,
                     node,
                     max_radii,
                     fire_temp,
                     air_temp):

    d1 = prop_df.loc[node]['d1']
    d2 = prop_df.iloc[node]['d2']
    d3 = prop_df.iloc[node]['d3']
    k = prop_df.iloc[node]['k']
    node_radii = prop_df.iloc[node]['radii']
    node_theta = prop_df.iloc[node]['theta']
    node_vf = prop_df.iloc[node]['node_vf']
    node_vol = prop_df.iloc[node]['volume']
    otr_area = prop_df.iloc[node]['otr_area']
    alpha = prop_df.iloc[node]['alpha']
    tock = tick + delta_time
    tick_temp = temp_df.loc[(tick, node)]['Temp[K]']

    if node_radii == max_radii:
        if (0 < node_theta) & (node_theta < math.pi):
            heat_flux = (node_vf * 5.67e-8 * (fire_temp ** 4 - tick_temp ** 4) +
                         (air_temp - tick_temp) * 15
                         ) * delta_time
            heat_gen = heat_flux * otr_area / node_vol
        else:
            heat_flux = (air_temp - tick_temp) * 15 * delta_time  # using an h-value of 15
            heat_gen = heat_flux * otr_area / node_vol
    else:
        heat_flux = 0
        heat_gen = 0

    inr_node = prop_df.iloc[node]['inr_nbr']

    if inr_node is np.nan:
        inr_temp = tick_temp
    else:
        inr_temp = temp_df.loc[(tick, inr_node)]['Temp[K]']

    lft_node = prop_df.iloc[node]['lft_nbr']
    lft_temp = temp_df.loc[(tick, lft_node)]['Temp[K]']

    rht_node = prop_df.iloc[node]['rht_nbr']
    rht_temp = temp_df.loc[(tick, rht_node)]['Temp[K]']

    otr_nodes = prop_df.iloc[node]['otr_nbr']

    if otr_nodes != list():
        otr_temp = 0
        for o_n in otr_nodes:
            otr_temp += temp_df.loc[(tick, o_n)]['Temp[K]']
        otr_temp /= len(otr_nodes)
    else:
        otr_temp = tick_temp

    tock_temp = (
            d1 * (inr_temp - 2 * tick_temp + otr_temp) +
            d2 * (inr_temp - otr_temp) +
            d3 * (rht_temp - 2 * tick_temp + lft_temp) +
            alpha * heat_gen / k +
            tick_temp
    )

    temp_df.loc[(tock, node)]['Temp[K]'] = tock_temp
    temp_df.loc[(tick, node)]['d1', 'd2', 'd3', 'k',
                              'alpha', 'tock_temp',
                              'heat_gen', 'heat_flux',
                              'otr_area',
                              'inr_temp', 'inr_nbr',
                              'otr_temp',
                              'lft_temp', 'lft_nbr',
                              'rht_temp', 'rht_nbr'] = [d1, d2, d3, k,
                                                        alpha, tock_temp,
                                                        heat_gen, heat_flux,
                                                        otr_area,
                                                        inr_temp, inr_node,
                                                        otr_temp,
                                                        lft_temp, lft_node,
                                                        rht_temp, rht_node]

    return temp_df


if __name__ == '__main__':
    try:
        pd.set_option('display.max_rows', 2000)
        pd.set_option('display.max_columns', 2000)
        pd.set_option('display.width', 2000)

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

        geo_prop_df = create_cyl_nodes(rings=inputs['rings'],
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

        geo_prop_df['c'] = geo_prop_df['comp'].apply(lambda cpnt: color_nodes_by_component(cpnt))

        if inputs['show_geo']:
            print('Building node visual...\n')
            generate_3d_node_geometry(prop_df=geo_prop_df)

        geo_prop_df = assign_node_view_factor(df=geo_prop_df, cyl_view_factor=vf)

        geo_prop_df = create_node_fdm_constants(geo_prop_df, comp_rhos, comp_Cps, comp_ks, inputs['TimeStep[s]'])

        time_steps = list(range(0, inputs['TimeIterations[#]'] + 1))
        time_steps = [i * inputs['TimeStep[s]'] for i in time_steps]
        time_steps = pd.Series(time_steps)

        nodes = list(geo_prop_df.index)

        idx = pd.MultiIndex.from_product([time_steps, nodes],
                                         names=['TimeStep', 'NodeIdx'])

        temp_df = pd.DataFrame(index=idx, columns=['Temp[K]'])

        temp_df['Temp[K]'] = inputs['Ambient/InitialTemp[C]'] + 273.15

        debug_columns = ['d1', 'd2', 'd3', 'k',
                         'alpha', 'tock_temp',
                         'heat_gen', 'heat_flux',
                         'otr_area',
                         'inr_temp', 'inr_nbr',
                         'otr_temp',
                         'lft_temp', 'lft_nbr',
                         'rht_temp', 'rht_nbr']

        for col in debug_columns:
            temp_df[col] = np.nan

        del debug_columns, idx, vf, tank_od, comp_ks, comp_Cps, comp_rhos

        otr_node_radii = geo_prop_df['radii'].max()
        fire_temp = inputs['FireTemp[C]'] + 273.15
        amb_temp = inputs['Ambient/InitialTemp[C]'] + 273.15

        print('Starting time iterations...')

        total_time_steps = int(len(time_steps)) - 1

        for t in time_steps[:-1]:
            print('\rCurrently on timestep {0} of {1}.'.format(int(t / inputs['TimeStep[s]']) + 1,
                                                               total_time_steps),
                  end='', flush=True)
            for n in nodes:
                temp_df = update_node_temp(prop_df=geo_prop_df,
                                           temp_df=temp_df,
                                           delta_time=inputs['TimeStep[s]'],
                                           tick=t,
                                           node=n,
                                           max_radii=otr_node_radii,
                                           fire_temp=fire_temp,
                                           air_temp=amb_temp)

                # temp_df = update_node_temp_attempt2(temp_df)

        print('\nFinished iterations.\n')

        print('Making graphics.\n')

        generate_time_gif(temp_df=temp_df, prop_df=geo_prop_df, time_steps=time_steps)

        generate_boundary_graphs(temp_df=temp_df,
                                 prop_df=geo_prop_df,
                                 time_steps=time_steps,
                                 features=['heat_flux',
                                           'tock_temp',
                                           'heat_gen'],
                                 labels=['Heat Flux\n[W/m²]',
                                         'Node\nTemperature\n[K]',
                                         'Heat\nGeneration\n[W/m³]'],
                                 color_map='hsv')

        print('\nFinished making graphics.\n')

        if export:
            print('Exporting results...\n')
            try:
                export_results(dfs=[geo_prop_df, temp_df],
                               df_names=['node_properties', 'node_temperatures'],
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






