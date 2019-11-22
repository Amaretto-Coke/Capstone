import shutil
import os
import subprocess
from time import strftime
import pandas as pd


def build_exe_version(script='',
                      description=''):
    os.chdir(os.path.dirname(os.getcwd()))
    if script == '':
        print('There was no script to build.')
        quit()
    version = make_version(script=script, description=description)

    generate_version_info(ver=version,
                          original_filename=script,
                          file_description=description)
    build_exe(script, no_console=False)
    exe_file = move_build(script)

    return exe_file


def generate_version_info(ver,
                          original_filename='ScriptName',
                          file_description=''):

    ver = [int(i) for i in ver]
    ver = [str(i) for i in ver]

    company_name = 'University of Calgary'
    legal_copyright = ' ' + company_name + '. All rights reserved.'
    product_name = 'FireRedTank_' + original_filename
    product_version = ver[0] + '.' + ver[1] + '.' + ver[2] + '.' + ver[3]
    file_version = product_version + ' (win7sp1_rtm.101119-1850)'

    lines = [
        'VSVersionInfo(\n',
        '  ffi=FixedFileInfo(\n',
        '    filevers=(' + ver[0] + ', ' + ver[1] + ', ' + ver[2] + ', ' + ver[3] + '),\n',
        '    prodvers=(' + ver[0] + ', ' + ver[1] + ', ' + ver[2] + ', ' + ver[3] + '),\n',
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
        '        [StringStruct(u\'CompanyName\', u\'' + company_name + '\'),\n',
        '        StringStruct(u\'FileDescription\', u\'' + file_description + '\'),\n',
        '        StringStruct(u\'FileVersion\', u\'' + file_version + ' (win7sp1_rtm.101119-1850)\'),\n',
        '        StringStruct(u\'InternalName\', u\'cmd\'),\n',
        '        StringStruct(u\'LegalCopyright\', u\'\\xa9 ' + legal_copyright + '\'),\n',
        '        StringStruct(u\'OriginalFilename\', u\'' + original_filename + '\'),\n',
        '        StringStruct(u\'ProductName\', u\'' + product_name + '\'),\n',
        '        StringStruct(u\'ProductVersion\', u\'' + product_version + '\')])\n',
        '      ]), \n',
        '    VarFileInfo([VarStruct(u\'Translation\', [1033, 1200])])\n',
        '  ]\n',
        ')',
    ]

    version_info = open(r'version.txt', 'w')

    for line in lines:
        version_info.write(line)

    version_info.close()


def make_version(script,
                 description,
                 new_rendition=False,
                 new_release=False,
                 new_edition=False):
    dtypes = {'Script': 'str',
              'Edition': 'int',
              'Release': 'int',
              'Rendition': 'int',
              'Build': 'int',
              'Version': 'float',
              'Description': 'str'}
    hist = pd.read_csv(r'BuildHistory.csv', index_col=0, dtype=dtypes)

    whist = hist[hist.Script == script]
    ver_list = ['', '', '', '']
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
    if new_edition or new_release or new_rendition:
        if new_rendition:
            ver_list[0] = whist.Edition[0]
            ver_list[1] = whist.Release[0]
            ver_list[2] = whist.Rendition[0] + 1
            ver_list[3] = whist.Build[0]
        elif new_release:
            ver_list[0] = whist.Edition[0]
            ver_list[1] = whist.Release[0] + 1
            ver_list[2] = whist.Rendition[0]
            ver_list[3] = whist.Build[0]
        else:  # new edition
            ver_list[0] = whist.Edition[0] + 1
            ver_list[1] = whist.Release[0]
            ver_list[2] = whist.Rendition[0]
            ver_list[3] = whist.Build[0]
    else:  # new build
        ver_list[0] = whist.Edition[0]
        ver_list[1] = whist.Release[0]
        ver_list[2] = whist.Rendition[0]
        ver_list[3] = whist.Build[0] + 1
    ver_list = [str(i) for i in ver_list]
    ver_list[1] = format(int(ver_list[1]), '02')
    data = {'Script': [script],
            'Edition': [ver_list[0]],
            'Release': [int(ver_list[1])],
            'Rendition': [ver_list[2]],
            'Build': [ver_list[3]],
            'Version': [float(''.join(ver_list))],
            'Description': [description]
            }
    new_entry = pd.DataFrame.from_dict(data=data, orient='columns').reset_index(drop=True)
    hist = hist.append(new_entry, sort=False).reset_index(drop=True)
    try:
        hist.to_csv(r'BuildHistory.csv')
    except PermissionError:
        print('History file not accessible, could not update Build History.')
        quit()
    return ver_list


def build_exe(py_script,
              one_file=True,
              no_console=True,
              icon=True,
              version=True):
    dist_path = ' --distpath=' + os.getcwd() + r'\dist'
    work_path = ' --workpath=' + os.getcwd() + r'\build'
    command = 'pyinstaller ' + os.path.abspath(os.getcwd() + r'\Scripts\\' + py_script + '.py') + dist_path + work_path
    if one_file:
        command += ' --onefile'
    if no_console:
        command += ' --noconsole'
    if icon:
        command += ' --icon=' + os.getcwd() + r'\icon.ico'
    if version:
        command += ' --version-file=' + os.getcwd() + r'\version.txt'
    command += ' --clean'
    try:
        subprocess.call(command)
    except FileNotFoundError:
        print('Could not run pyinstaller from the command prompt.')
        quit()


def move_build(script=''):
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

    build_outputs = [r'\\build\\', r'\\dist', r'\\' + script + '.spec']

    for Output in build_outputs:
        shutil.move(src=os.getcwd() + Output, dst=target_folder + Output)

    for file in os.listdir(target_folder + r'build\\' + script):
        source = target_folder + r'build\\' + script + r'\\' + file
        shutil.move(src=source, dst=target_folder + r'build\\')

    shutil.rmtree(target_folder + r'build\\' + script + r'\\')

    source = target_folder + r'dist\\' + script + '.exe'

    shutil.move(src=source, dst=target_folder)

    shutil.rmtree(target_folder + r'dist\\')

    return target_folder + r'\\' + script + '.exe'
