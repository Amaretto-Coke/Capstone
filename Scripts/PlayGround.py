import pandas as pd

df = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3', 'A4'],
                   'B': ['B0', 'B1', 'B2', 'B3', 'B4'],
                   'C': ['C0', 'C1', 'C2', 'C3', 'C4'],
                   'D': ['D0', 'D1', 'D2', 'D3', 'D4'],
                   'E': ['E0', 'E1', 'E2', 'E3', 'E4'],
                   'F': ['F0', 'F1', 'F2', 'F3', 'F4'],
                   'Z': [1, 1, 2, 3, 0]
                   })

print(df)

x = df.loc[df['Z'], 'A']

print(x)

x.reset_index(drop=True, inplace=True)

print(x)

