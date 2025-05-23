# MLEyetrack_fast.spec
# -*- mode: python; coding: utf-8 -*-

FAST_BUILD = True # False = single .exe file

from PyInstaller.utils.hooks import collect_all

# gather everything OpenCV needs
datas_cv2, binaries_cv2, hiddenimports_cv2 = collect_all('cv2')
datas_tf2onnx, binaries_tf2onnx, hiddenimports_tf2onnx = collect_all('tf2onnx')
datas_onnxruntime, binaries_onnxruntime, hiddenimports_onnxruntime = collect_all('onnxruntime')

if FAST_BUILD:
    block_cipher = None

    a = Analysis(
        ['MLEyetrack.py'],
        pathex=['.'],           # point at your script folder
        binaries=binaries_cv2 + binaries_tf2onnx + binaries_onnxruntime,
        datas=datas_cv2 + datas_tf2onnx + datas_onnxruntime,
        hiddenimports=['colorama', 'tensorflow'] + hiddenimports_cv2 + hiddenimports_tf2onnx + hiddenimports_onnxruntime,
        hookspath=[],
        runtime_hooks=['OpenCVPatch.py'],
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