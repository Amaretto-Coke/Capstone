# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:31:52 2020

@author: MaeganV
"""
import sys
import traceback
from SetUp import *
from Graphics import *
from PostOffice import *
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
from Solver import *

path = os.path.dirname(os.getcwd()) + r'\ParameterStudy.xlsx'
parameterstudy = pd.read_excel(path, sheet_name='readme')
parameterstudy.set_index('CaseName', drop=True, inplace=True)
cases = list(parameterstudy.index)
            
for i in cases:
    print('Running case ' + i)
    inputs = import_cases_and_fluids()
    key = parameterstudy.KeyIndex[i]
    param_value = parameterstudy.Property[i]
    inputs[key] = param_value
    
    try:
        if True:
            pd.set_option('display.max_rows', 2000)
            pd.set_option('display.max_columns', 2000)
            
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
            
            dumpDF = node_df.copy()
            
            for t in time_steps[:-1]:
                radiation = node_df.copy(deep=True)
                conv = node_df.copy(deep=True)
                radiation.drop(radiation[radiation.theta > np.pi/2].index, inplace=True)
                conv.drop(conv[conv.theta < np.pi/2].index, inplace=True)
                netheat = round(sum(node_df.iloc[:,-3].to_numpy()),4)
                netrad = round(sum(radiation.iloc[:,-3].to_numpy()),4)
                netconv = round(sum(conv.iloc[:,-3].to_numpy()),4)
                
                print('\rTotal Heat Flux (Radiation+Convection) at time {0}s of {1}s is {2}+{3}={4} W/m2.'.format(
                int(t) + 1, total_time_steps,netrad,netconv,netheat),
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
                
                if t == int(inputs['TimeIterations[#]']/2):
                    filename = 'node_df_' + i + '_part1.pkl'
                    node_df.to_pickle(filename)
                    results = node_df.iloc[:,-3:]
                    results.reset_index(drop=True, inplace=True)
                    del node_df
                    node_df = pd.concat([dumpDF, results], axis=1)
                    
    
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
        
    filename = 'node_df_' + i + '.pkl'
    node_df.to_pickle(filename)