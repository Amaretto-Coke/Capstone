# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['C:\\Users\\Brand\\PycharmProjects\\Capstone\\Scripts\\Solver.py'],
             pathex=['C:\\Users\\Brand\\PycharmProjects\\Capstone'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Solver',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , version='C:\\Users\\Brand\\PycharmProjects\\Capstone\\version.txt', icon='C:\\Users\\Brand\\PycharmProjects\\Capstone\\icon.ico')
