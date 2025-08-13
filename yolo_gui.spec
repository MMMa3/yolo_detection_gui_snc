# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 分析主脚本
a = Analysis(
    ['yolo_gui.py'],
    pathex=['d:\\PythonProjects\\yolo_detection_gui_snc'],
    binaries=[],
    datas=[
        # 包含yolov5整个目录
        ('yolov5', 'yolov5'),
        # 包含图标文件
        ('icon.ico', '.'),
        # 包含输出目录（如果存在）
        ('output', 'output'),
    ],
    hiddenimports=[
        # PyTorch相关
        'torch',
        'torchvision',
        'torch.nn',
        'torch.nn.functional',
        'torch.optim',
        'torch.utils',
        'torch.utils.data',
        'torchvision.transforms',
        'torchvision.models',
        
        # OpenCV相关
        'cv2',
        'cv2.dnn',
        
        # 科学计算库
        'numpy',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        
        # 图像处理
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        
        # 绘图库
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends',
        'matplotlib.backends.backend_tkagg',
        'seaborn',
        
        # YAML处理
        'yaml',
        
        # 其他工具库
        'requests',
        'tqdm',
        'pathlib',
        'threading',
        'io',
        'sys',
        'os',
        'argparse',
        'typing',
        
        # YOLOv5特定模块
        'models',
        'models.common',
        'models.experimental',
        'models.yolo',
        'utils',
        'utils.general',
        'utils.torch_utils',
        'utils.plots',
        'utils.metrics',
        'utils.dataloaders',
        'utils.augmentations',
        'utils.autoanchor',
        'utils.autobatch',
        'utils.callbacks',
        'utils.downloads',
        'utils.loss',
        'utils.activations',
        
        # Tkinter相关（虽然通常内置，但为了确保兼容性）
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        
        # 其他可能需要的模块
        'psutil',
        'gitpython',
        'git',
        'ultralytics',
        'thop',
        'pandas',
        'setuptools',
        'logging.config',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除一些不需要的模块以减小体积
        'tensorboard',
        'tensorflow',
        'tensorflowjs',
        'onnx',
        'coremltools',
        'openvino',
        'tritonclient',
        'clearml',
        'comet_ml',
        'wandb',
        'neptune',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 收集所有文件
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 创建可执行文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='YoloDetectApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 设置为False以隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',  # 设置应用程序图标
)

# 收集所有依赖文件
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='YoloDetectApp',
)

# 如果需要创建单文件版本，可以取消注释以下部分：
# exe = EXE(
#     pyz,
#     a.scripts,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     [],
#     name='YOLO_Detection_GUI',
#     debug=False,
#     bootloader_ignore_signals=False,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     runtime_tmpdir=None,
#     console=False,
#     disable_windowed_traceback=False,
#     argv_emulation=False,
#     target_arch=None,
#     codesign_identity=None,
#     entitlements_file=None,
#     icon='icon.ico',
# )