import pandas as pd
import matplotlib.pyplot as plt
import math
#import os
#from PostOffice import *


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


def get_df_value(df, level, col, row):
    df_levs = sorted(list(set(df.index.get_level_values(0))))
    df_cols = sorted(list(set(df.index.get_level_values(1))))
    df_rows = sorted(list(set(df.index.get_level_values(2))))

    if level in df_levs:
        pass
    elif level > max(df_levs):
        level -= max(df_levs) + 1
    elif level < min(df_levs):
        level += max(df_levs) - 1
    else:
        print('Level', level, 'not in, above, or below df levels.')

    if col in df_cols:
        pass
    elif col > max(df_cols):
        col -= max(df_cols) + 1
    elif col < min(df_cols):
        col += max(df_cols) - 1
    else:
        print('Column', col, 'not in, above, or below df columns.')

    if row in df_rows:
        pass
    elif row > max(df_rows):
        row -= max(df_rows) + 1
    elif row < min(df_rows):
        row += max(df_rows) - 1
    else:
        print('Row', row, 'not in, above, or below df rows.')

    return df.loc[(level, col, row)].Loc





if __name__=='__main__':
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
''' # 2D Nodal Position Plot

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
                print(get_df_value(df=df, level=level+1, col=col+1, row=row+1))

    df.reset_index(inplace=True)


