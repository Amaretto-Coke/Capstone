import sys
import traceback
import time
from SetUp import *
from Graphics import *
from PostOffice import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero


def update_node_temp_pd(time_step):
    pass

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



        if inputs['show_geo']:
            print('Building node visual...\n')
            generate_3d_node_geometry(prop_df=node_df)

        node_df = assign_node_view_factor(df=node_df, cyl_view_factor=vf)

        node_df = create_node_fdm_constants(node_df, comp_rhos, comp_Cps, comp_ks, inputs['TimeStep[s]'])

        time_steps = list(range(0, inputs['TimeIterations[#]'] + 1))
        time_steps = ["t={:0.2f}s".format(i * inputs['TimeStep[s]']) for i in time_steps]

        nodes = list(node_df.index)
        init_temp = np.float64(inputs['Ambient/InitialTemp[C]'] + 273.15)
        node_df = node_df.assign(**{i: init_temp for i in time_steps})

        otr_node_radii = node_df['radii'].max()
        fire_temp = np.float64(inputs['FireTemp[C]'] + 273.15)
        amb_temp = np.float64(inputs['Ambient/InitialTemp[C]'] + 273.15)

        print('Starting time iterations...')

        total_time_steps = int(len(time_steps)) - 1

        for col in node_df.columns.to_list():
            print(col)
        time.sleep(2)
        quit()

        for t in time_steps[:-1]:
            print('\rCurrently on timestep {0} of {1}.'.format(
                int(t / inputs['TimeStep[s]']) + 1, total_time_steps),
                  end='', flush=True)
            for n in nodes:
                pass

        print('\nFinished iterations.\n')

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

        if export:
            print('Exporting results...\n')
            try:
                export_results(dfs=[node_df],
                               df_names=['node_temperatures'],
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


