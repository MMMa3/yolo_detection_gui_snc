#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动YOLO检测GUI界面的便捷脚本
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'tkinter',
        'PIL',
        'torch',
        'cv2',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'PIL':
                from PIL import Image
            elif package == 'torch':
                import torch
            elif package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements_gui.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 YOLODetectionGUI")
    print("=" * 50)
    
    # 检查依赖
    print("\n📦 检查依赖包...")
    if not check_dependencies():
        input("\n按回车键退出...")
        return
    
    # 检查yolo_predict.py文件
    yolo_predict_path = Path(__file__).parent / 'yolov5' / 'yolo_predict.py'
    if not yolo_predict_path.exists():
        print(f"❌ 找不到YOLO预测脚本: {yolo_predict_path}")
        print("请确保yolo_predict.py文件在正确的位置")
        input("\n按回车键退出...")
        return
    
    print("✅ YOLO预测脚本已找到")
    
    # 启动GUI
    try:
        print("\n🎯 启动GUI界面...")
        from yolo_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ 启动GUI失败: {str(e)}")
        print("\n请检查以下事项:")
        print("1. 确保所有依赖包已正确安装")
        print("2. 确保yolo_predict.py文件存在且可访问")
        print("3. 检查Python版本兼容性")
        input("\n按回车键退出...")

if __name__ == '__main__':
    main()