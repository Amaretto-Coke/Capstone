import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cb
import numpy as np
import seaborn as sns
import math, warnings, datetime
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=cb.mplDeprecation)
pd.set_option('mode.chained_assignment', None)
register_matplotlib_converters()

# Function to lookup the well name in the dictionary and assign a Tag to it.
def AssignWellTag(wellName):
    try:
        return wellDict[wellName]
    except KeyError:
        print('Error assigning ', wellName)
        pass

# Defining a function to use with scaling the y-axis later
def sizeNum(num):
    itr = 1
    while num > 10:
        num /= 10
        itr += 1
    return num, itr

# A function to get the text positions and return new ones where they don't overlap.
# Used when annotating on the graph.
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):

        local_text_positions = [i for i in a if i[0] > (y - txt_height) and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

# A function to annotate the text on their new positions on the chosen plot.
def text_plotter(x_data, text_positions,txt,  axis, txt_width, txt_height, withArrows=False):
    if withArrows:
        for x, y, t in zip(x_data, text_positions, txt):
            axis.text(x - txt_width, 1.01 * y, '%d' % int(t), rotation=0, color='black')
            if y != t:
                axis.arrow(x, t, 0, y-t, color='black',alpha=0.3, width=txt_width*0.1,
                           head_width=txt_width, head_length=txt_height*0.5,
                           zorder=0,length_includes_head=True)
    else:
        for x,y,t in zip(x_data, text_positions, txt):
            axis.text(x, 1.01*y, '%d'%int(t),rotation=0, color='black')


# Colour codes:
Grass    = '#78C850' # 120 200  80
Fire     = '#F08030' # 240 128  48
Water    = '#6890F0'
Bug      = '#A8B820'
Normal   = '#A8A878'
Poison   = '#A040A0'
Electric = '#F8D030'
Ground   = '#E0C068'
Fairy    = '#EE99AC'
Fighting = '#C03028'
Psychic  = '#F85888'
Rock     = '#B8A038'
Ghost    = '#705898'
Ice      = '#98D8D8'
Dragon   = '#7038F8'
Steel    = '#B8B8D0' # 184 184 208
Dark     = '#705848' # 112  88  72
Unknown  = '#68A090' # 104 160 144
Flying   = '#A890F0' # 168 144 240

# Retrieving the data
print('Loading Volumes from OFM Export...')
vol_df   = pd.read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\OFM_Volumes.xlsx', index_col=0)
print('Loading Parameters from Zafiro Export...')
para_df  = pd.read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\Zafiro_Parameters.xlsx')
print('Loading Test Dates from Zafiro Export...')
tests_df = pd.read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\Zafiro_Tests.xlsx')
print('Loading events and zone history from Excel files...')
event_df = pd.read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\EventHistory.xlsx')
zone_df  = pd.read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\ZonalHistory.xlsx')

# Load in the Well Dictionary
wellDict = pd \
    .read_excel(r'N:\PutumayoNorthWellHistory\~DataBank\WellDictionary.xlsx') \
    .set_index('WellNames', drop=True) \
    .T \
    .to_dict(orient='records')[0]


print('Setting common identifiers and manipulating the data...\n')

# Identify the wells by the parent wellbore
vol_df  ['Tag'] = vol_df  ['OFM_Wellbore'].apply(lambda x: AssignWellTag(x))
para_df ['Tag'] = para_df ['Well Hole'].apply(lambda x: AssignWellTag(x))
tests_df['Tag'] = tests_df['Well Hole'].apply(lambda x: AssignWellTag(x))

# Relabeling the 'Operative Date' in the Zafiro Data Frames to 'Date'
para_df ['Date'] = para_df ['Operative Date']
tests_df['Date'] = tests_df['Operative Date']
para_df = para_df.drop(columns=['Operative Date', 'Well Hole', 'Wells', 'Well'])
tests_df = tests_df.drop(columns=['Operative Date', 'Well Hole', 'Wells', 'Well'])

# Getting a List of all the Zones in the Zone History
zoneList = zone_df.columns[4:]
zone_df['diff_days'] = (zone_df['EndDate'] - zone_df['StartDate']) / np.timedelta64(1, 'D') + 1

# Setting the test dates to 100 so it appears as a vertical line on the Water Cut percentage graph.
tests_df['Test'] = 100

# Resets the index of the data frame so that it starts at 0 again
zone_df = zone_df.reset_index(drop=True)

# Consolidating the Zones volumes into Wellbore volumes
vol_df = vol_df[['Date', 'Tag', 'Gas [Mcft]', 'Oil [bbls]', 'Water Prod. [bbls]', 'Water Injected [bbls]']] \
    .groupby(['Date', 'Tag']).sum()
vol_df['Index'] = range(0, len(vol_df))
vol_df = vol_df.reset_index().drop(columns='Index')

print('Generating Calculated Variables...\n')

# Calculating secondary variables
vol_df['GOR [scf/bbl]'] = vol_df['Gas [Mcft]'] * 1000 / vol_df['Oil [bbls]']
vol_df['WCut [%]'] = vol_df['Water Prod. [bbls]'] / (vol_df['Oil [bbls]'] + vol_df['Water Prod. [bbls]']) * 100
vol_df['WCut [%]'] = vol_df['WCut [%]'] \
    .fillna(value=0) \
    .replace([np.inf, -np.inf], 0)
vol_df['Liquid Produced [bbl/d]'] = vol_df['Oil [bbls]'] + vol_df['Water Prod. [bbls]']

print('Merging the Data...\n')

# Merging the Data Frames
all_df = vol_df.merge(event_df, on=['Date', 'Tag'], how='outer')
all_df = all_df.merge(para_df, on=['Date', 'Tag'], how='outer')
all_df = all_df.merge(tests_df, on=['Date', 'Tag'], how='outer')

# Filling NAN in some columns with 0
all_df = all_df.fillna(value=0)

# Setting terminal dataframe print() settings
pd.options.display.max_rows = 500
pd.options.display.max_columns = 999
pd.options.display.width = 0

WellList = all_df['Tag'] \
    .drop_duplicates() \
    .sort_values() \
    .reset_index(drop=True) \
    .tolist()

gridSpace = (11, 3)
x_max = datetime.date(2019, 8, 12)

print('Looping through the wellbores...')
# Looping through the wells in the data frame, creating a plot per wellbore
for well in WellList:
    '''
    del well
    well = 'CYC-15'
    '''
    print('\n\t' + well, 'in progress...')

    # Creating a working data frame that only includes our current wells data
    well_df = all_df[all_df['Tag'] == well]

    x_min_list = [
        well_df[well_df['Test'] == 100]['Date'].min(),
        well_df[well_df['Oil [bbls]'] > 0]['Date'].min(),
        well_df[well_df['Gas [Mcft]'] > 0]['Date'].min(),
        well_df[well_df['Water Prod. [bbls]'] > 0]['Date'].min(),
        well_df[well_df['Water Injected [bbls]'] > 0]['Date'].min(),
        well_df[well_df['Event'] == 1]['Date'].min(),
    ]

    x_min_list = [i for i in x_min_list if not pd.isnull(i)]
    try:
        x_min = min(x_min_list)
    except ValueError:  # aka if x_min_list is empty
        x_min = datetime.date(2009, 1, 1)

    # Creation of a letter paper-sized figure
    fig = plt.figure(figsize=(16.5, 11.7), dpi=100)  # figsize=(11.69,8.27)

    plt.xticks(rotation=25)

    # Adding a title
    fig.suptitle(well, fontsize=16)

    # Setting the space between subplots
    plt.subplots_adjust(hspace=0, wspace=0.3)

    # Creation of the first subplot (main graph)
    ax0 = plt.subplot2grid(gridSpace, (0, 0), colspan=3, rowspan=4)

    # Plotting the Water Cut percentage.
    line5 = sns.lineplot(x='Date', y='WCut [%]', data=well_df, ax=ax0, color=Water, label='Water Cut', alpha=1)
    # Plotting the Well Tests
    line4 = sns.lineplot(x='Date', y='Test', data=well_df, ax=ax0, color=Steel, label='Well Tests',
                         alpha=0.25)  # 0.25
    # Filling in underneath the line
    ax0.fill_between(x=well_df['Date'].values, y1=well_df['Test'].values, color=Steel, alpha=0.25)  # 0.1

    # Formatting the plot
    ax0.set_xlabel('Date')
    ax0.set_ylabel('Water Cut [%]')
    ax0.set_ylim(bottom=0, top=100)  # top=ax0.get_yticks()[-1]
    ax0.tick_params(labelbottom=False, top=True, bottom=True)
    ax0.set_xlim(x_min, x_max)

    # Create a working copy of the event history data frame that only includes our wells data
    wehdf = event_df[event_df['Tag'] == well]
    wehdf['PointToPlot'] = 1

    # Underlaying the well test graph (ax0)
    ax1 = ax0.twinx()

    # Plotting Oil, Total Liquid and GOR, and Water Injected Data if it's available.
    line1 = sns.lineplot(x='Date', y='GOR [scf/bbl]', data=well_df, ax=ax1, color=Fighting, label='GOR')
    line2 = sns.lineplot(x='Date', y='Liquid Produced [bbl/d]', data=well_df, ax=ax1, color=Unknown, label='Liquid')
    line3 = sns.lineplot(x='Date', y='Oil [bbls]', data=well_df, ax=ax1, color=Grass, label='Oil')

    # Creates the annotation plot points on the graph.
    adf = well_df[well_df['Event'] > 0].reset_index(drop=True)
    adf['Event'] = adf['Event'].apply(lambda x: int(x))

    # Annotates the Well Event Numbers to the Graph
    eventDates = adf['Date'].to_list()
    eventNumber = adf['Event'].to_list()
    eventPoints = adf['PlotPoint'].to_list()
    eventComments = adf['Comment'].to_list()

    # Adjusts the numbers so that there's no overlap.
    txt_height = 0.04 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    anno_x_max = datetime.datetime.combine(x_max, datetime.datetime.min.time())
    anno_x_min = datetime.datetime.combine(x_min, datetime.datetime.min.time())
    txt_width = 0.02 * (anno_x_max - anno_x_min)
    eventPoints = get_text_positions(x_data=eventDates, y_data=eventPoints, txt_width=txt_width, txt_height=txt_height)
    text_plotter(x_data=eventDates, text_positions=eventPoints, axis=ax1, txt=eventNumber, txt_width=txt_width, txt_height=txt_height)

    # Setting the bounds for the top of the main graph, ax1.
    # Setting it to the maximum Liquid Produced value, rounded up to (10^x)/2,
    # where x is the order of magnitude of the maximum.
    # For example, for a maximum of 2561 the graph y maximum would be 3000.
    my_max = well_df['Liquid Produced [bbl/d]'].max()
    ymax, yscale = sizeNum(my_max)
    ymax = math.ceil(ymax * 2) / 2 * 10 ** (yscale - 1)
    yinterval = 10 ** (yscale - 1) / 2
    if ymax > 10000:
        ymax = 10000
        yinterval = 2500

    # Setting ticks for the first Graph (aka 'ax'), in intervals of half the order of magnitude
    # from 0 to the chart maximum.
    yticks = []
    tck = 0
    while tck < ymax:
        yticks.append(tck)
        tck += yinterval
    yticks.append(ymax)

    ax1.set_yticks(yticks)
    ax1.set_ylabel('Liquid and Oil (bbl/d); GOR (scf/bbl)')
    ax1.set_ylim(bottom=0, top=ymax)
    ax1.set_title('Oil, Total Liquid; Gas-Oil Ratio')
    ax1.tick_params(labelbottom=False, top=True, bottom=True)
    ax1.legend().remove()

    lines = ax0.lines + ax1.lines
    labels = [l.get_label() for l in ax0.lines] + [l.get_label() for l in
                                                   ax1.lines]  # + [l.get_label() for l in ax2.lines]
    ax0.legend(lines, labels, loc='upper center', facecolor='w', ncol=len(labels) + 1)
    ax1.set_zorder(ax0.get_zorder() + 1)  # put ax1 in front of ax0

    # Creation of the second and third subplots (ax3 & ax4), which are overlayed to form graph 2
    ax3 = plt.subplot2grid(gridSpace, (4, 0), colspan=3, rowspan=2, sharex=ax1)

    if well_df['PIP'].sum() > 0:
        # Plotting the Pump Intake Pressure
        sns.lineplot(x='Date', y='PIP', data=well_df, ax=ax3, color=Poison,
                     label='Pump Intake Pressure [psi]')
        # Formatting the plot
        ax3.set_ylabel('[psi]')
        ax3.legend(loc='upper left', facecolor='w')
        ax3.set_ylim(bottom=0, top=2000)  # Ensures the top of the graph is on a tick mark ax3.get_yticks()[-1]
        ax3.set_yticks(ax3.get_yticks()[0:-1])  # Setting the ticks to omit the top label
        ax3.tick_params(labelbottom=False, top=True, bottom=True)
    else:
        ax3.set_yticks([])

    # Adding the third subplot
    ax4 = ax3.twinx()  # Share the x ax with the previous plot
    # sns.lineplot(x='Date', y='Power Fl. BPD', data=well_df, ax=ax3, color=Bug, label='Power Fluid [bbl/d]')
    if well_df['Frequencia_Bomba (Hz)'].sum() > 0:
        sns.lineplot(x='Date', y='Frequencia_Bomba (Hz)', data=well_df, ax=ax4, color=Ghost,
                     label='Pump Frequency [Hz]')
        # Formating the plot
        ax4.set_ylabel('[Hz]')
        ax4.set_ylim(bottom=0)  # top=ax3.get_yticks()[-1]
        ax4.legend(loc='upper right', facecolor='w')
        ax4.set_yticks(ax4.get_yticks()[0:-1])
        ax4.tick_params(labelbottom=False, top=True, bottom=True)
    else:
        ax4.set_yticks([])

    # Creating the Zone History plot:
    ax6 = plt.subplot2grid(gridSpace, (6, 0), colspan=3, rowspan=2, sharex=ax1)
    tdf = zone_df[zone_df['Tag'] == well]
    yticklabels = []
    yticks = []
    i = 0
    for zone in zoneList:
        myzdf = tdf[tdf[zone] == 1].reset_index(drop=True)
        if sum(myzdf[zone]) > 0:
            for indexNum, row in myzdf.iterrows():
                ax6.broken_barh([(row['StartDate'], row['diff_days'])], (0 + i, 9), facecolors=Rock)
            i += 10
            yticklabels.append(zone)
            yticks.append(i - 5.5)
    ax6.set_yticks(yticks)
    ax6.set_yticklabels(yticklabels)

    ax6.set_ylabel(' Zone History')
    plt.xticks(rotation=25, ha='right')

    # Creating the Event History notes space:
    ax7 = plt.subplot2grid(gridSpace, (9, 0), colspan=3, rowspan=2)
    if adf.shape[0] > 0:
        numOfEvents = adf['Event'].max()
        adf['ChartOrder'] = numOfEvents - adf['Event']
        my_min = well_df['Date'].min()
        ax7.set_ylim(top=numOfEvents + 1, bottom=-1)
        ax7.set_xlim(left=my_min)
        for i, txt in enumerate(eventNumber):
            ax7.annotate(
                ' ' + str(txt) + ': ' + eventComments[i],
                (my_min, adf['ChartOrder'][i]),
                fontsize=int(adf['FontSize'][i])
            )
    ax7.set_yticks([], [])
    ax7.set_xticks([], [])
    ax7.set_ylabel(' Event History')

    #plt.show()
    # break # if you only want it to loop through the first well (for testing).

    try:
        # Saving the plot to a letter sized paper PDF
        fig.savefig(r'N:\PutumayoNorthWellHistory\~ProductionWellGraphsOutput\\' + well + '_ProductionGraph.pdf',
                    edgecolor='k',
                    papertype='legal',
                    orientation='landscape',
                    dpi=100,
                    format='pdf',
                    #pad_inches=0,
                    #bbox_inches='tight',
                    )

    except PermissionError:
        print('\tCould not save', well, 'to PDF, check to see if file is open,\nand if so, close it then resume.')
        answer = str(input('Continue? Y/N\n'))
        if not (answer == 'y' or answer == 'Y'):
            quit()
    finally:
        pass

    # Clearing the figure to save up memory
    fig.clf()
    plt.close(fig)

    print('\t' + well, 'complete.')

print('Graphing complete.')