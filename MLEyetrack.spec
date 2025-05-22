# MLEyetrack_fast.spec
# -*- mode: python; coding: utf-8 -*-

FAST_BUILD = True # False = single .exe file

if FAST_BUILD:
    block_cipher = None

    a = Analysis(
        ['MLEyetrack.py'],
        pathex=['.'],           # point at your script folder
        binaries=[],
        datas=[],
        hiddenimports=['colorama', 'cv2', 'tensorflow'],
        hookspath=[],
        runtime_hooks=[],
        excludes=[],
        noarchive=True,          # skip bundling pure modules into a .pyz
        optimize=0,              # no byte-code optimization step
    )
    # include zipped_data even though we’re not archiving
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name='MLEyetrack',
        debug=False,
        strip=False,             # don’t run strip on the bootloader
        upx=False,               # disable UPX to avoid its compression time
        console=True,
        icon='./images/deprivedlogo_transparentandwhitebackground.ico'
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,               # again, ensure no UPX on collected files
        name='MLEyetrack',
    )
else:
    pass