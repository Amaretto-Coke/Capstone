from Scripts.ExeFactory import *

description = 'Testing Build'# input('Build description for ' + Script + ':\n')

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

exe = build_exe_version(script=Scripts[ans], description='An application to run the ProductionPlotter.py code.')

print('Build successful:', Scripts[ans] + '.exe')

os.system('start ' + exe)
