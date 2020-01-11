'''

from math import exp
from math import log
import pandas as pd

print('[A]')


def bitumen_viscosity_cP(temp_C):
    return exp(exp(23.168-3.630*log(temp_C+273.15))) - 0.7


def solvent_viscosity_cP(temp_C, solvent):
    table = {'Propane': [-3.45, 415.4],
             'Butane':  [-3.65, 556.6],
             'Pentane': [-3.84, 697.9],
             'Hexane':  [-4.04, 839.1],
             'Heptane': [-4.23, 980.4]}

    a, b = table[solvent]

    return exp(a + b / (temp_C + 273.15))


solvents = ['Propane', 'Butane', 'Pentane', 'Hexane', 'Heptane']

viscosity = bitumen_viscosity_cP(10)
print('The viscosity for Bitumen at 10°C is: {0} cP'.format('{:.3f}'.format(viscosity)))

for solvent in solvents:
    viscosity = solvent_viscosity_cP(10, solvent)
    print('The viscosity for {0} at 10°C is: {1} cP'.format(solvent, '{:.3f}'.format(viscosity)))
print('\n')

print('[B]')
print('The solvent that should be used is Propane, since it has the lowest viscosity.')
print('\n')

print('[C]')


def solvent_fraction(mixture_viscosity_cP,
                     solvent_viscosity_cP,
                     bitumen_viscosity_cP):

    numerator = log(mixture_viscosity_cP) - log(bitumen_viscosity_cP)
    denominator = log(solvent_viscosity_cP) - log(bitumen_viscosity_cP)

    return numerator/denominator


mol_fractions = [0]
for solvent in solvents:
    sol_vis_cP = solvent_viscosity_cP(10, solvent)
    bit_vis_cP = bitumen_viscosity_cP(10)
    sol_fraction = solvent_fraction(mixture_viscosity_cP=5,
                                    solvent_viscosity_cP=sol_vis_cP,
                                    bitumen_viscosity_cP=bit_vis_cP)
    print('The solvent mol fraction for {0} at 10°C is: {1}.'.format(solvent, '{:.4f}'.format(sol_fraction)))
    mol_fractions.append(sol_fraction)
print('\n')

print('[D]')


def bitumen_density(temp_C):
    return 1024 - 0.645 * temp_C

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)

data = {'Solvent': ['Ethane', 'Propane', 'Butane', 'Pentane', 'Hexane', 'Heptane'],
        'T_C': [32.27, 96.67, 152.03, 196.5, 234.3, 267.1],
        'P_C': [4.88, 4.25, 3.797, 3.369, 3.012, 2.736],
        'MW': [30, 44, 58, 72, 86, 100],
        'Z_RA': [.2789, .2763, .2728, .2685, .2635, .2611],
        'Z_C': [.284, .280, .274, .262, .264, .263],
        'V_a': [148, 203, 255, 304, 370, 432],
        'ω': [.0908, .1454, .1928, .251, .2657, .3506],
        'sol_mol_fraction': mol_fractions
        }

table2 = pd.DataFrame(data=data).set_index('Solvent', drop=True)

table2['T_R'] = (10 + 273.15)/(table2['T_C'] + 273.15)

table2['sol_ρ'] = table2.MW * (table2.P_C * 1000) / 8.314 / (table2.T_C + 273.15) * table2.Z_RA**((1 + (1 - table2.T_R)**(2/7)) * -1)

table2['bit_ρ'] = bitumen_density(10)

table2['sol_V'] = table2.sol_mol_fraction * table2.MW/1000 * table2.sol_ρ

table2['bit_V'] = (1 - table2.sol_mol_fraction) * 500/1000 * table2.bit_ρ

table2['sol_vol_fraction'] = table2.sol_V / (table2.sol_V + table2.bit_V)

print(table2.head(6))
print('\n')

print('[E]')
print('Propane, since the least amount of propane is required per m³ of bitumen.')
'''

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

'''
def points_all_on_plane(p1, p2, p3, p4):
    """
    Evaluates 4 coordinates to determine if they are on the
        same plane in 3D space.
    :param p1: The first    point, as a 1 x 3 numpy array.
    :param p2: The second   point, as a 1 x 3 numpy array.
    :param p3: The third    point, as a 1 x 3 numpy array.
    :param p4: The forth    point, as a 1 x 3 numpy array.
    :return: True or False, depending on if all 4 points are on a plane.
    """

    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p1

    cp1 = np.cross(v1, v2)
    cp2 = np.cross(v1, v3)

    cp1 = normalize_vector(cp1)
    cp2 = normalize_vector(cp2)

    result = (cp1[0] == cp2[0]) and (cp1[1] == cp2[1]) and (cp1[2] == cp2[2])

    return result
'''  # Points on plane function

my_file = r'C:\\Users\\Brand\OneDrive - University of Calgary\\Capstone\\GradedDocuments\\Scripts-2019_12_08.pdf'


from pdf2image import convert_from_path
pages = convert_from_path(my_file, 500)

i = 1
for page in pages:
    page.save('Page{0}.jpg'.format('{:.2}'.format(i)), 'JPEG')
    i += 1