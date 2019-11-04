import pandas as pd
import os
from PostOffice import *


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



''' 
Testing Data Export Function
data = [['chris', 10], ['nick', 15], ['juli', 14]]

df1 = pd.DataFrame(data=data)
df2 = pd.DataFrame(data=data)

dfs = [df1, df2]

export_results(dfs, ['df1', 'df2'])

path = os.path.dirname(os.getcwd()) + r'\MailBox.xlsx'

os.startfile(path)

'''