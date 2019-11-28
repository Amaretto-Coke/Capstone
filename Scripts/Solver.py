import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
import seaborn as sns
import sys
from PostOffice import *
from SetUp import *
from Graphics import *


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
            heat_flux = (node_vf * 5.67e-8 * (fire_temp**4 - tick_temp**4) +
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

    if otr_nodes != []:
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


def update_node_temp_attempt2(temp_df):
    pass


if __name__ == '__main__':
    try:
        pd.set_option('display.max_rows', 2000)
        pd.set_option('display.max_columns', 2000)
        pd.set_option('display.width', 2000)

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
        '''  # 2D Nodal Position Plot.
    
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
        '''  # Temperature Dataframe in hierarchical indexing.
    
        '''
        geo_prop_df = vf_plane_to_cyl1(s=20, r=10, l=10, t=20, n=100)
    
        print(geo_prop_df)
    
        geo_prop_df = vf_plane_to_cyl2(s=20, r=10, l=10, t=20, n=100)
    
        print(geo_prop_df)
        '''  # Comparison of View Factor Formulas.
    
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
    
        geo_prop_df = points_all_on_plane(p1=p1, p2=p2, p3=p3, p4=p4)
    
        print(geo_prop_df)
        '''  # Testing for vector normalization function and is_plane function.
        
        '''
        from PostOffice import *
    
        geo_prop_df = create_cyl_nodes(rings=3,
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
            export_results(dfs=[geo_prop_df], df_names=['Testing'], open_after=True, index=True)
        except PermissionError:
            print('File is locked for editing by user.\nNode network could not be exported.')
    
        geo_prop_df['c'] = geo_prop_df['comp'].apply(lambda cpnt: color_nodes_by_component(cpnt))
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(geo_prop_df.x.to_list(),
                   geo_prop_df.y.to_list(),
                   geo_prop_df.z.to_list(),
                   c=geo_prop_df.c.to_list(),
                   s=5)
        ax.set_axis_off()
        plt.show()
        
        '''  # Testing the node geometry creation functions.
        """  # Older Testing

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

        tank_od = inputs['TankID[m]'] + inputs['WallThickness[cm]']/100

        vf = vf_plane_to_cyl2(s=inputs['FireDistanceFromTankCenter[m]'],
                              r=tank_od/2,
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
                                       wall_thickness=inputs['WallThickness[cm]']/100)

        geo_prop_df['c'] = geo_prop_df['comp'].apply(lambda cpnt: color_nodes_by_component(cpnt))

        if inputs['show_geo']:
            print('Building node visual...\n')
            generate_3d_node_geometry(prop_df=geo_prop_df)
            exit()

        geo_prop_df = assign_node_view_factor(df=geo_prop_df, cyl_view_factor=vf)

        geo_prop_df = create_node_fdm_constants(geo_prop_df, comp_rhos, comp_Cps, comp_ks, inputs['TimeStep[s]'])

        time_steps = list(range(0, inputs['TimeIterations[#]']+1))
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
                         'rht_temp', 'rht_nbr' ]

        for col in debug_columns:
            temp_df[col] = np.nan

        del debug_columns, idx, vf, tank_od, comp_ks, comp_Cps, comp_rhos

        otr_node_radii = geo_prop_df['radii'].max()
        fire_temp = inputs['FireTemp[C]'] + 273.15
        amb_temp = inputs['Ambient/InitialTemp[C]'] + 273.15

        print('Starting time iterations...')

        total_time_steps = int(len(time_steps))-1

        for t in time_steps[:-1]:
            print('\rCurrently on timestep {0} of {1}.'.format(int(t/inputs['TimeStep[s]'])+1,
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
        import sys
        print(sys.exc_info()[0])
        import traceback
        print(traceback.format_exc())
    finally:
        print("Press Enter to continue ...")
        input()






