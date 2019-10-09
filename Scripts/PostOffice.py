import pandas as pd
import os
import math


def import_cases_and_fluids():
    # Creating a path string to the user interface excel file
    path = os.path.dirname(os.getcwd()) + r'\MailBox.xlsm'

    # Importing the two Excel sheets as two new data frames
    cases = pd.read_excel(path, sheet_name='Cases')
    fluid_properties = pd.read_excel(path, sheet_name='FluidProperties')

    # Deleting the units from the column headers
    cases.columns = [col[0:col.find('[')] for col in cases.columns]

    # Creating two new columns, of the inside cylinder volume and the initial liquid volume
    cases['CylinderVolume'] = math.pi * (cases.TankID / 2) ** 2 * cases.TankHeight
    cases['LiquidVolume'] = math.pi * (cases.TankID / 2) ** 2 * cases.FluidLevel

    # Merges the fluid properties into the cases dataframe
    cases = cases.merge(fluid_properties, on='FluidName', how='left')

    # Clears the now useless fluid properties dataframe
    del fluid_properties

    # Returns a the merged dataframe
    return cases


def export_results(dfs=None):
    # Creating a path string to the user interface excel file
    path = os.path.dirname(os.getcwd()) + r'\MailBox.xlsm'

    # Creating a writer for the pandas to_excel function
    writer = pd.ExcelWriter(path)

    for df in dfs:
        df.to_excel(writer, sheet_name=df.name)
        
        
def validate_excel_sheet_name(val_string):
    not_allowed = ['/', '*', '?', ':', '[', ']', '\\']
