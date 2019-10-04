import pandas as pd
import os
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

cases = pd.read_excel(os.path.dirname(os.getcwd()) + r'\MailBox.xlsm', sheet_name='Cases')
fluid_properties = pd.read_excel(os.path.dirname(os.getcwd()) + r'\MailBox.xlsm', sheet_name='Fluid Properties')

# Deleting the units from the column headers
cases.columns = [col[0:col.find('[')] for col in cases.columns]

cases['CylinderVolume'] = math.pi * (cases.TankID / 2) ** 2 * cases.TankHeight
cases['FluidVolume'] = math.pi * (cases.TankID / 2) ** 2 * cases.FluidLevel

print(cases)

