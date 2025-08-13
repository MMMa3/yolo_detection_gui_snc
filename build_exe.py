#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动打包脚本
使用PyInstaller将YOLO GUI应用程序打包为可执行文件
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import PyInstaller

def check_pyinstaller():
    """检查PyInstaller是否已安装"""
    try:
        print(f"✓ PyInstaller已安装，版本: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("✗ PyInstaller未安装")
        print("请运行: pip install pyinstaller")
        return False

def clean_build_dirs():
    """清理之前的构建目录"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"清理目录: {dir_name}")
            shutil.rmtree(dir_name)

def build_executable():
    """构建可执行文件"""
    spec_file = 'yolo_gui.spec'
    
    if not os.path.exists(spec_file):
        print(f"✗ 找不到spec文件: {spec_file}")
        return False
    
    print("开始构建可执行文件...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        # 运行PyInstaller
        cmd = [sys.executable, '-m', 'PyInstaller', spec_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 构建成功！")
            print(f"可执行文件位置: dist/YoloDetectApp/")
            return True
        else:
            print("✗ 构建失败")
            print("错误信息:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ 构建过程中出现异常: {e}")
        return False

def create_single_file():
    """创建单文件版本（可选）"""
    print("\n是否要创建单文件版本？(y/n): ", end='')
    choice = input().lower().strip()
    
    if choice == 'y' or choice == 'yes':
        print("创建单文件版本...")
        try:
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--onefile',
                '--windowed',
                '--icon=icon.ico',
                '--name=YoloDetectApp_Single',
                '--add-data=yolov5;yolov5',
                '--add-data=icon.ico;.',
                '--add-data=output;output',
                'yolo_gui.py'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ 单文件版本创建成功！")
                print(f"单文件位置: dist/YOLO_Detection_GUI_Single.exe")
            else:
                print("✗ 单文件版本创建失败")
                print(result.stderr)
                
        except Exception as e:
            print(f"✗ 创建单文件版本时出现异常: {e}")

def main():
    """主函数"""
    print("YOLO GUI 应用程序打包工具")
    print("=" * 40)
    
    # 检查PyInstaller
    if not check_pyinstaller():
        return
    
    # 清理构建目录
    print("\n清理之前的构建文件...")
    clean_build_dirs()
    
    # 构建可执行文件
    print("\n开始构建...")
    if build_executable():
        print("\n构建完成！")
        print("\n使用说明:")
        print("1. 可执行文件位于 dist/YoloDetectApp/ 目录中")
        print("2. 运行 YoloDetectApp.exe 启动应用程序")
        print("3. 首次运行时需要加载YOLO模型文件(.pt)")
        print("4. 确保将整个 dist/YoloDetectApp/ 目录复制到目标机器")

        # 询问是否创建单文件版本
        create_single_file()
        
    else:
        print("\n构建失败，请检查错误信息并重试")

if __name__ == '__main__':
    main()