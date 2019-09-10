import shutil, os, subprocess
from time import strftime
from pathlib import Path
import pandas as pd


def BuildExeVersion(Script='', Description=''):
    os.chdir(os.path.dirname(os.getcwd()))
    if Script == '':
        print('There was no script to build.')
        quit()
    Version = MakeVersion(script=Script, description=Description)

    GenerateVersionInfo(Ver=Version,
                        OriginalFilename=Script,
                        FileDescription=Description)
    BuildExe(Script, noconsole=False)
    exeFile = MoveBuild(Script)
    return exeFile

def GenerateVersionInfo(Ver, OriginalFilename='ScriptName', FileDescription=''):

    Ver = [int(i) for i in Ver]
    Ver = [str(i) for i in Ver]

    CompanyName = 'Gran Tierra Energy'
    LegalCopyright = ' ' + CompanyName + '. All rights reserved.'
    ProductName = 'Emerald_' + OriginalFilename
    ProductVersion = Ver[0] + '.' + Ver[1] + '.' + Ver[2] + '.' + Ver[3]
    FileVersion = ProductVersion + ' (win7sp1_rtm.101119-1850)'

    Lines = [
        'VSVersionInfo(\n',
        '  ffi=FixedFileInfo(\n',
        '    filevers=(' + Ver[0] + ', ' + Ver[1] + ', ' + Ver[2] + ', ' + Ver[3] + '),\n',
        '    prodvers=(' + Ver[0] + ', ' + Ver[1] + ', ' + Ver[2] + ', ' + Ver[3] + '),\n',
        '    mask=0x3f,\n',
        '    flags=0x0,\n',
        '    OS=0x40004,\n',
        '    fileType=0x1,\n',
        '    subtype=0x0,\n',
        '    date=(0, 0)\n',
        '    ),\n',
        '  kids=[\n',
        '    StringFileInfo(\n',
        '      [\n',
        '      StringTable(\n',
        '        u\'040904B0\',\n',
        '        [StringStruct(u\'CompanyName\', u\'' + CompanyName + '\'),\n',
        '        StringStruct(u\'FileDescription\', u\'' + FileDescription + '\'),\n',
        '        StringStruct(u\'FileVersion\', u\'' + FileVersion + ' (win7sp1_rtm.101119-1850)\'),\n',
        '        StringStruct(u\'InternalName\', u\'cmd\'),\n',
        '        StringStruct(u\'LegalCopyright\', u\'\\xa9 ' + LegalCopyright + '\'),\n',
        '        StringStruct(u\'OriginalFilename\', u\'' + OriginalFilename + '\'),\n',
        '        StringStruct(u\'ProductName\', u\'' + ProductName + '\'),\n',
        '        StringStruct(u\'ProductVersion\', u\'' + ProductVersion + '\')])\n',
        '      ]), \n',
        '    VarFileInfo([VarStruct(u\'Translation\', [1033, 1200])])\n',
        '  ]\n',
        ')',
    ]

    versionInfo = open(r'version.txt', 'w')

    for line in Lines:
        versionInfo.write(line)

    versionInfo.close()

def MakeVersion(script, description, newRendition=False, newRelease=False, newEdition=False):
    dtypes = {'Script': 'str',
              'Edition': 'int',
              'Release': 'int',
              'Rendition': 'int',
              'Build': 'int',
              'Version': 'float',
              'Description': 'str'}
    hist = pd.read_csv(r'BuildHistory.csv', index_col=0, dtype=dtypes)
    whist = hist[hist.Script == script]
    verList = ['', '', '', '']
    try:
        whist = whist[whist.Version == max(whist.Version)].reset_index(drop=True)
    except ValueError:  # For if data frame is empty
        whist = whist.append({'Script': script,
                              'Edition': 0,
                              'Release': 0,
                              'Rendition': 0,
                              'Build': 0,
                              'Version': 0,
                              'Description': 'Zeroth Build'},
                             ignore_index=True)
        pass
    if newEdition or newRelease or newRendition:
        if newRendition:
            verList[0] = whist.Edition[0]
            verList[1] = whist.Release[0]
            verList[2] = whist.Rendition[0] + 1
            verList[3] = whist.Build[0]
        elif newRelease:
            verList[0] = whist.Edition[0]
            verList[1] = whist.Release[0] + 1
            verList[2] = whist.Rendition[0]
            verList[3] = whist.Build[0]
        else:  # new edition
            verList[0] = whist.Edition[0] + 1
            verList[1] = whist.Release[0]
            verList[2] = whist.Rendition[0]
            verList[3] = whist.Build[0]
    else:  # new build
        verList[0] = whist.Edition[0]
        verList[1] = whist.Release[0]
        verList[2] = whist.Rendition[0]
        verList[3] = whist.Build[0] + 1
    verList = [str(i) for i in verList]
    verList[1] = format(int(verList[1]), '02')
    data = {'Script': [script],
            'Edition': [verList[0]],
            'Release': [int(verList[1])],
            'Rendition': [verList[2]],
            'Build': [verList[3]],
            'Version': [float(''.join(verList))],
            'Description': [description]
            }
    newEntry = pd.DataFrame.from_dict(data=data, orient='columns').reset_index(drop=True)
    hist = hist.append(newEntry, sort=False).reset_index(drop=True)
    try:
        hist.to_csv(r'BuildHistory.csv')
    except PermissionError:
        print('History file not accessable, could not update Build History.')
        quit()
    return verList

def BuildExe(pyScript, onefile=True, noconsole=True, icon=True, version=True):
    distpath = ' --distpath=' + os.getcwd() + r'\dist'
    workpath = ' --workpath=' + os.getcwd() + r'\build'
    command = 'pyinstaller ' + os.path.abspath(os.getcwd() + r'\Scripts\\' + pyScript + '.py') + distpath + workpath
    if onefile:
        command += ' --onefile'
    if noconsole:
        command += ' --noconsole'
    if icon:
        command += ' --icon=' + os.getcwd() + r'\icon.ico'
    if version:
        command += ' --version-file=' + os.getcwd() + r'\version.txt'
    command += ' --clean'
    subprocess.call(command)

def MoveBuild(script=''):
    if script == '':
        print('Script not specified.')
        quit()

    time = strftime("%Y%m%d-%H%M%S")

    # Creates the out path as a string
    builds_folder = os.getcwd().replace("\\", r'\\') + r'\\' + script + '_Builds'
    target_folder = builds_folder + r'\\' + time + r'\\'

    if not os.path.exists(builds_folder):
        os.makedirs(builds_folder)

    os.makedirs(target_folder)

    BuildOutputs = [r'\\build\\', r'\\dist', r'\\' + script + '.spec']

    for Output in BuildOutputs:
        shutil.move(src=os.getcwd() + Output, dst=target_folder + Output)

    for file in os.listdir(target_folder + r'build\\' + script):
        source = target_folder + r'build\\' + script + r'\\' + file
        shutil.move(src=source, dst=target_folder + r'build\\')

    shutil.rmtree(target_folder + r'build\\' + script + r'\\')

    source = target_folder + r'dist\\' + script + '.exe'

    shutil.move(src=source, dst=target_folder)

    shutil.rmtree(target_folder + r'dist\\')

    return target_folder + r'\\' + script + '.exe'

Scripts = [Path(f).stem for f in os.listdir(os.getcwd())]
scriptDict = {Scripts.index(s) : s for s in Scripts}
print("{:<8} {:<15}".format('Index','Script'))
for k, v in scriptDict.items():
    print("{:<8} {:<10}".format(k,v))


ans = ''
inIndex = True
while isinstance(ans, str) or not inIndex:
    ans = input('\nEnter index of script to build, or enter x to terminate.\nIndex must be an integer.\n')
    if ans == 'x':
        print('Script terminated.')
        try:
            exit()
        except NameError:
            sys.exit()
    try:
        ans = int(ans)
        inIndex = 0 <= ans < len(Scripts)
    except ValueError:
        pass

print('Building', Scripts[ans]+'.py')

exe = BuildExeVersion(Script=Scripts[ans], Description='An application to run the ProductionPlotter.py code.')

print('Build successful:', Scripts[ans] + '.exe')

os.system('start ' + exe)
#print(os.path.dirname(exe))
#subprocess.call('explorer ' + os.path.dirname(exe), shell=True)

