#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试颜色功能的脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from yolo_predict import YOLOPredictor

def test_color_generation():
    """测试颜色生成功能"""
    print("测试颜色生成功能...")
    
    # 创建一个简单的预测器实例来测试颜色方法
    class TestPredictor:
        def _get_class_color(self, class_id: int) -> tuple:
            """为不同类别生成对比度高的颜色"""
            # 预定义一些对比度高的颜色 (BGR格式)
            colors = [
                (0, 255, 0),     # 绿色
                (255, 0, 0),     # 蓝色  
                (0, 0, 255),     # 红色
                (255, 255, 0),   # 青色
                (255, 0, 255),   # 品红色
                (0, 255, 255),   # 黄色
                (128, 0, 128),   # 紫色
                (255, 165, 0),   # 橙色
                (0, 128, 255),   # 橙红色
                (255, 20, 147),  # 深粉红色
                (0, 255, 127),   # 春绿色
                (255, 105, 180), # 热粉红色
                (64, 224, 208),  # 青绿色
                (255, 69, 0),    # 红橙色
                (50, 205, 50),   # 绿黄色
                (138, 43, 226),  # 蓝紫色
                (255, 140, 0),   # 深橙色
                (72, 61, 139),   # 深板岩蓝
                (220, 20, 60),   # 深红色
                (0, 206, 209),   # 深青绿色
            ]
            
            # 如果类别数超过预定义颜色数，使用算法生成颜色
            if class_id < len(colors):
                return colors[class_id]
            else:
                # 使用HSV色彩空间生成颜色，确保高饱和度和亮度
                import colorsys
                hue = (class_id * 137.508) % 360  # 使用黄金角度确保颜色分布均匀
                rgb = colorsys.hsv_to_rgb(hue / 360, 0.9, 0.9)  # 高饱和度和亮度
                return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # 转换为BGR
    
    predictor = TestPredictor()
    
    # 测试前20个类别的颜色
    print("前20个类别的颜色 (BGR格式):")
    for i in range(20):
        color = predictor._get_class_color(i)
        print(f"类别 {i}: {color}")
    
    print("\n测试超出预定义范围的类别:")
    for i in range(20, 25):
        color = predictor._get_class_color(i)
        print(f"类别 {i}: {color}")
    
    print("颜色测试完成！")

if __name__ == '__main__':
    test_color_generation()
