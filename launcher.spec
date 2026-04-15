# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None


def collect_folder_files(source_root, target_root):
    source_root = Path(source_root)
    entries = []
    for file_path in source_root.rglob("*"):
        if not file_path.is_file():
            continue
        relative_parent = file_path.relative_to(source_root).parent
        target_dir = Path(target_root) / relative_parent
        entries.append((str(file_path), str(target_dir)))
    return entries

datas = [
    *collect_folder_files("runtime", "runtime"),
    *collect_folder_files("model\\simplifier-4090", "model\\simplifier-4090"),
    ("translate.py", "."),
    *collect_data_files("streamlit"),
    *copy_metadata("streamlit"),
]

hiddenimports = [
    "streamlit.web.bootstrap",
    "streamlit.web.cli",
    "streamlit.runtime.scriptrunner",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.runtime.state",
    "streamlit.delta_generator",
    "streamlit",
    "torch",
    "peft",
    "transformers",
    "IndicTransToolkit",
    "bitsandbytes",
    "sentencepiece",
    "sacremoses",
    "huggingface_hub",
    "accelerate",
    "safetensors",
    "einops",
    "numpy",
    "pandas",
    "tqdm",
    "requests",
    "regex",
    "scipy",
]

# Transformers resolves many model families lazily at runtime; include model submodules
# so Auto* loaders do not fail in the frozen executable.
hiddenimports += collect_submodules("transformers.models")

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["gguf", "full"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TextSimplifierRuntime",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name="TextSimplifierRuntime",
)