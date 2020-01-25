import sys
import traceback
from SetUp import *
from Graphics import *
from PostOffice import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


if __name__ == '__main__':
    pd.set_option('display.max_rows', 2000)
    # pd.set_option('display.max_columns', 2000)
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

    geo_prop_df = geo_prop_df[geo_prop_df['radii'] == geo_prop_df['radii'].max()]
    geo_prop_df = geo_prop_df[geo_prop_df['theta'] >= 0]
    geo_prop_df = geo_prop_df[geo_prop_df['theta'] <= math.pi]

    #if inputs['show_geo']:
    if True:
        print('Building node visual...\n')
        generate_3d_node_geometry(prop_df=geo_prop_df)

    geo_prop_df = assign_node_view_factor(df=geo_prop_df, cyl_view_factor=vf)

    geo_prop_df = create_node_fdm_constants(geo_prop_df, comp_rhos, comp_Cps, comp_ks, inputs['TimeStep[s]'])

    fire_temp = inputs['FireTemp[C]'] + 273.15
    amb_temp = inputs['Ambient/InitialTemp[C]'] + 273.15
    h_values = list(range(0, 31, 3))
    h_values = [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]
    wall_temps = list(range(270, 1300, 5))

    #  h_values = [i * inputs['TimeStep[s]'] for i in h_values]
    h_values = pd.Series(h_values)

    nodes = list(geo_prop_df.index)

    idx = pd.MultiIndex.from_product([h_values, wall_temps, nodes],
                                     names=['h_value[W/m²K]', 'Wall_Temp[K]', 'NodeIdx'])

    temp_df = pd.DataFrame(index=idx, columns=['Heat_Flux[W/m²]'])

    temp_df['theta'] = np.nan

    del idx, tank_od, comp_ks, comp_Cps, comp_rhos

    print('Starting h_val iterations...')
    for h in h_values:
        print('On h value {0}.'.format(h))
        for temp in wall_temps:
            hf = vf * 5.67e-8 * (fire_temp ** 4 - temp ** 4) - (temp - amb_temp) * h
            if hf < 0:
                break
            for n in nodes:
                temp_df = update_node_temp(prop_df=geo_prop_df,
                                           node_temp=temp,
                                           node=n,
                                           fire_temp=fire_temp,
                                           air_temp=amb_temp,
                                           h_val=h)

            # temp_df = update_node_temp_attempt2(temp_df)

    print('\nFinished iterations.\n')

    ave_df = temp_df.reset_index()
    ave_df['Heat_Flux[W/m²]'] = ave_df['Heat_Flux[W/m²]'].astype('float64')
    ave_df = ave_df.groupby(['h_value[W/m²K]', 'Wall_Temp[K]']).mean()
    ave_df.reset_index(inplace=True)
    ave_df.drop(['theta', 'NodeIdx'], axis=1, inplace=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for h in h_values:
        line_df = ave_df[ave_df['h_value[W/m²K]'] == h]
        ax.plot(line_df['Wall_Temp[K]'], line_df['Heat_Flux[W/m²]'], label=h)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_xlabel('Cyl_Wall_Temp[K]')
    ax.set_ylabel('Heat_Flux[W/m²]')
    ax.legend(loc='best', title='h value [W/m²K]')
    ax.set_title('Cylinder Wall Temperature [K] vs. Net Heat Flux Absorbed [W/m²],\nwith different values of h')
    plt.show()

    '''
    print('Making graphics.\n')

    generate_time_gif(temp_df=temp_df, prop_df=node_df, time_steps=h_values)

    generate_boundary_graphs(temp_df=temp_df,
                             prop_df=node_df,
                             time_steps=h_values,
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
            temp_df.reset_index(inplace=True)
            export_results(dfs=[geo_prop_df, temp_df, ave_df],
                           df_names=['node_properties', 'node_temperatures', 'average_heat_flux'],
                           open_after=True,
                           index=False)
        except PermissionError:
            print('File is locked for editing by user.\n\tNode network could not be exported.')