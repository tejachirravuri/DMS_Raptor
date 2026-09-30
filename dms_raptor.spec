# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for DMS-Raptor."""

import os
import sys

block_cipher = None

# Paths
APP_DIR = os.path.dirname(os.path.abspath(SPEC))
RESOURCES = os.path.join(APP_DIR, 'resources')

a = Analysis(
    ['main.py'],
    pathex=[APP_DIR],
    binaries=[],
    datas=[
        (os.path.join(RESOURCES, 'style.qss'), 'resources'),
        (os.path.join(RESOURCES, 'style_light.qss'), 'resources'),
        (os.path.join(RESOURCES, 'logo.svg'), 'resources'),
        (os.path.join(RESOURCES, 'logo.png'), 'resources'),
        (os.path.join(RESOURCES, 'logo.ico'), 'resources'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtSvg',
        'cv2',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_qtagg',
        'ultralytics',
        'torch',
        'torchvision',
        'pandas',
        'docx', 'docx.shared', 'docx.enum.text', 'docx.enum.table',
        'docx.enum.style', 'docx.oxml', 'docx.oxml.ns',
        'gui.thesis_tab', 'gui.thesis_plots', 'gui.thesis_worker',
        'gui.analysis_tab', 'gui.analysis_worker',
        'core.analysis_engine', 'core.analysis_runner', 'core.csv_logger',
        'core.informed_gain', 'core.controllers', 'core.policies',
        'core.matching', 'core.metrics', 'core.proxies',
        'core.inference_backend',
        'scripts.generate_thesis_report',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DMS-Raptor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(RESOURCES, 'logo.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DMS-Raptor',
)
