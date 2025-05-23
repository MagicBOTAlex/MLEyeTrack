# MLEyetrack_fast.spec
# -*- mode: python; coding: utf-8 -*-
FAST_BUILD = True  # True = directory build; False = single-file .exe

# 1. Collect everything for these packages in one loop
packages = ['cv2', 'tf2onnx', 'onnxruntime', 'onnxruntime-gpu', 'tensorflow']
datas = []
binaries = []
hiddenimports = []

from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
import os
base_nvidia_path = os.path.join(os.environ.get("CONDA_PREFIX", ""), "Lib", "site-packages", "nvidia")
subdirs = ["cudnn", "cublas", "cuda_runtime", "cufft", "nvjitlink"]

for subdir in subdirs:
    bin_path = os.path.join(base_nvidia_path, subdir, "bin")
    if os.path.isdir(bin_path):
        for dll in os.listdir(bin_path):
            if dll.lower().endswith(".dll"):
                binaries.append((os.path.join(bin_path, dll), "."))

for _ in range(10):
    print()
print("Base nvidia:", base_nvidia_path)
    
for pkg in packages:
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

hiddenimports += ['colorama', 'tensorflow']

# How the fuck does python have this much bloat?!?!
# print("\n")
# print(datas)
# print("\n")
# print(binaries)
# print("\n")
# print(hiddenimports)
# print("\n")

for _ in range(10):
    print()

block_cipher = None

if FAST_BUILD:
    # directory build (no onefile)
    print("Making fast build")
    a = Analysis(
        ['MLEyetrack.py'],
        pathex=['.'],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        runtime_hooks=['OpenCVPatch.py'],
        excludes=[],
        noarchive=True,
        optimize=0,
    )
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name='MLEyetrack',
        debug=False,
        strip=False,
        upx=False,
        console=True,
        icon='./images/deprivedlogo_transparentandwhitebackground.ico'
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='MLEyetrack',
    )

else:
    # one-file build
    print("Making singular .exe build")
    
    import PyInstaller.config
    PyInstaller.config.CONF['distpath'] = "./dist/MLEyetrack/"
    
    a = Analysis(
        ['MLEyetrack.py'],
        pathex=['.'],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        hookspath=[],
        runtime_hooks=['OpenCVPatch.py'],
        excludes=[],
        noarchive=False,
        optimize=0,
    )
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='MLEyetrack',
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='./images/deprivedlogo_transparentandwhitebackground.ico'
    )
