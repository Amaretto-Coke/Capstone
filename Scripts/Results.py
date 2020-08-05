# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:44:21 2020

@author: MaeganV
"""
from PostOffice import export_results, import_cases_and_fluids
import pandas as pd
import numpy as np
from Graphics import createheatmap, outernodegraph, innernodegraph
import os


def resultsgetter(inputs,node_df):
    columns = ['InnerNodes_min','InnerNodes_ave','InnerNodes_max',
               'WallNodes_min','WallNodes_ave','WallNodes_max',
               'Wall_UpperNodes_min','Wall_UpperNodes_ave','Wall_UpperNodes_max']
    
    results_df = pd.DataFrame(np.zeros([1,9]),columns = columns)
    
    if inputs['Mode'] == 'Steady_State':
        createheatmap(node_df,inputs['Case_name'],0,inputs['Mode'])
        
        #get inner node values at end time
        insidetemp = node_df[node_df.radii < node_df.radii.max()]
        insidetemp = insidetemp['Temp']
        
        #get wall node values at end time
        outsidetemp = node_df[node_df.radii == node_df.radii.max()]
        outsidetemp_l = outsidetemp[outsidetemp['node_class'] == 5]['Temp']
        outsidetemp_u = outsidetemp[outsidetemp['node_class'] == 6]['Temp']
        
    elif inputs['Mode'] == 'Fixed_Time':
        time_steps = list(range(0, inputs['TimeIterations[#]'] + 1))  #
        str_time_steps = ["t={:0.2f}s".format(
                i * inputs['TimeStep[s]']) for i in time_steps]
        times = ['T @ ' + i for i in str_time_steps]
        
        #get inner node values at end time
        insidetemp = node_df[node_df.radii < node_df.radii.max()][times[-1]]
        
        #get wall node values at end time
        outsidetemp = node_df[node_df.radii == node_df.radii.max()]
        outsidetemp_l = outsidetemp[outsidetemp['node_class'] == 5][times[-1]]
        outsidetemp_u = outsidetemp[outsidetemp['node_class'] == 6][times[-1]]

    results_df.InnerNodes_ave = insidetemp.mean()
    results_df.InnerNodes_min = insidetemp.min()
    results_df.InnerNodes_max = insidetemp.max()
    
    results_df.WallNodes_ave = outsidetemp_l.mean()
    results_df.WallNodes_min = outsidetemp_l.min()
    results_df.WallNodes_max = outsidetemp_l.max()
    
    results_df.Wall_UpperNodes_ave = outsidetemp_u.mean()
    results_df.Wall_UpperNodes_min = outsidetemp_u.min()
    results_df.Wall_UpperNodes_max = outsidetemp_u.max()

    print('Exporting results...')
    try:
           export_results(dfs=[results_df],
           df_names=[inputs['Case_name']],
           open_after=False,
           index=True)
           print('Results sucuessfully exported')
           
    except PermissionError:
            print('File is locked for editing by user.\nResults could not be exported.')
          
    #create plts
    print('Creating results plots...')
    if inputs['Mode'] == 'Steady_State':
        createheatmap(node_df,inputs['Case_name'],0,inputs['Mode'])
        
    elif inputs['Mode'] == 'Fixed_Time':  
        createheatmap(node_df,inputs['Case_name'],times[-1],inputs['Mode'])
        outernodegraph(node_df[node_df['node_class']==5],time_steps,str_time_steps,inputs['Case_name'])
        outernodegraph(node_df[node_df['node_class']==6],time_steps,str_time_steps,inputs['Case_name']+'_UpperNodes')
        innernodegraph(node_df[node_df['comp']=='Liquid'],time_steps,str_time_steps,inputs['Case_name'])
        
    print('Plots sucuessfully created')


 #set up for exporting results
inputs = import_cases_and_fluids()
path = os.getcwd() + r'\node_df_' + inputs['Case_name'] + r'.pkl'
node_df = pd.read_pickle(path)
resultsgetter(inputs,node_df)