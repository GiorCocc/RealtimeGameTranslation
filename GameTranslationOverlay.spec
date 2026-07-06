# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [('settings.json', '.')]
binaries = []
hiddenimports = []

# collect_all() raccoglie insieme datas + binaries + submoduli (hiddenimports)
# di un pacchetto. A differenza di collect_data_files() (usato in precedenza),
# include anche i binari nativi (.pyd/.dll) e gli import dinamici che
# PaddleX/PaddleOCR usano per il discovery dei moduli a runtime — la causa
# più comune di "funziona da IDE ma crasha nel build" con questi pacchetti.
for pkg in ("paddlex", "paddleocr", "paddle"):
    _datas, _binaries, _hiddenimports = collect_all(pkg)
    datas += _datas
    binaries += _binaries
    hiddenimports += _hiddenimports

datas += copy_metadata('paddleocr')
datas += copy_metadata('paddlex')

# Dipendenze usate condizionalmente da PaddleOCR/PaddleX/EasyOCR che
# l'analisi statica a volte non rileva (import lazy o dinamici).
hiddenimports += [
    "skimage",
    "shapely",
    "shapely.geometry",
    "pyclipper",
    "lmdb",
    "imgaug",
    "yaml",
    "PIL",
    "easyocr",
]

a = Analysis(
    ['run_beta.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # scikit-learn (e quindi scipy.stats) viene importato SOLO da una
    # feature opzionale di transformers (candidate_generator.py, per la
    # "assisted generation") che il progetto non usa (si genera con
    # num_beams=1 diretto). scipy.stats ha un bug noto che causa
    # 'NameError: name obj is not defined' in ambiente frozen/PyInstaller
    # (docstring del modulo non popolata come atteso). Escluderlo qui
    # evita che rientri silenziosamente nel bundle se in futuro un'altra
    # dipendenza lo reinstalla nel venv.
    excludes=['sklearn'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GameTranslationOverlay',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # IMPORTANTE per la fase beta: console=True mantiene stdout/stderr
    # reali invece di None. Con --windowed (console=False), qualunque
    # print/warning emesso da Paddle/PaddleOCR durante l'init può causare
    # 'NoneType' object has no attribute 'write' PRIMA che app.log sia
    # configurato — un crash silenzioso che sembra "legato a Paddle" ma
    # in realtà è solo l'assenza di uno stream valido.
    console=True,
    # Se hai PyInstaller >= 6.11 puoi ottenere il meglio di entrambi i mondi:
    # una console che esiste (niente None) ma che si nasconde dopo l'avvio
    # riuscito. Decommenta la riga sotto e imposta console=False sopra:
    # hide_console='hide-late',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GameTranslationOverlay',
)
