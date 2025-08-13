#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO预测脚本 - 参数化版本
支持单张图片、多张图片文件夹、网络视频流预测
返回JPEG二进制流结果
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import io
from typing import Union, List, Optional

# 添加yolov5路径到系统路径
sys.path.append(str(Path(__file__).parent / 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_coords

class YOLOPredictor:
    """YOLO预测器类"""
    
    def __init__(self, model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.5):
        """
        初始化YOLO预测器
        
        Args:
            model_path: 模型权重文件路径
            conf_thres: 置信度阈值
            iou_thres: NMS阈值
        """
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None  # 存储类别名称
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = DetectMultiBackend(self.model_path, device=self.device)
            # 获取类别名称
            self.class_names = self.model.names if hasattr(self.model, 'names') else None
            print(f"模型加载成功，使用设备: {self.device}")
            if self.class_names:
                print(f"检测到 {len(self.class_names)} 个类别")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
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
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 调整图像大小
        img = cv2.resize(image, (640, 480))
        # 转换为tensor并归一化
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor.to(self.device)
    
    def _postprocess_detections(self, pred, img_tensor_shape, original_shape):
        """后处理检测结果"""
        detections = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        
        results = []
        for det in detections:
            if len(det):
                # 将坐标缩放回原始图像尺寸
                det[:, :4] = torch.from_numpy(
                    scale_coords(img_tensor_shape[2:], det[:, :4].cpu().numpy(), original_shape)
                ).to(det.device).type(det.dtype).round()
                
                # 转换为numpy数组
                det_np = det.cpu().numpy()
                for *xyxy, conf, cls in det_np:
                    results.append({
                        'bbox': [int(x) for x in xyxy],  # [x1, y1, x2, y2]
                        'confidence': float(conf),
                        'class': int(cls)
                    })
        
        return results
    
    def _draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """在图像上绘制检测结果"""
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class']
            
            # 根据类别获取颜色
            color = self._get_class_color(cls)
            
            # 绘制边界框 - 使用类别特定颜色
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # 绘制中心点 - 使用更深的颜色
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # 将颜色调暗作为中心点颜色
            dark_color = tuple(int(c * 0.7) for c in color)
            cv2.circle(annotated_image, (center_x, center_y), 4, dark_color, -1)

            # 添加标签 - 使用类别名称而不是序号
            if self.class_names and cls < len(self.class_names):
                class_name = self.class_names[cls]
                label = f"{class_name} {conf:.2f}"
            else:
                label = f"cls:{cls} conf:{conf:.2f}"
            
            # 计算标签背景尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制标签背景矩形 - 使用类别特定颜色
            padding = 5
            label_y = max(y1 - 10, text_height + padding)  # 确保标签不会超出图像顶部
            cv2.rectangle(annotated_image, 
                         (x1, label_y - text_height - padding),
                         (x1 + text_width + padding * 2, label_y + baseline),
                         color, -1)
            
            # 绘制标签文字 - 使用白色或黑色，根据背景颜色自动选择
            # 计算颜色亮度来决定文字颜色
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            
            cv2.putText(annotated_image, label, (x1 + padding, label_y - padding), 
                       font, font_scale, text_color, thickness)
        
        return annotated_image
    
    def predict_image(self, image: np.ndarray) -> tuple:
        """预测单张图像并返回JPEG二进制流和检测结果"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 预处理
        img_tensor = self._preprocess_image(image)
        
        # 推理
        with torch.no_grad():
            pred = self.model(img_tensor)
        
        # 后处理
        detections = self._postprocess_detections(pred, img_tensor.shape, image.shape)
        
        # 绘制检测结果
        annotated_image = self._draw_detections(image, detections)
        
        # 如果没有检测到任何对象，在图像中央添加"non-detected"标签
        if not detections:
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # 添加"non-detected"文本
            text = "non-detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 3
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 计算文本位置（居中）
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            # 绘制背景矩形
            padding = 10
            cv2.rectangle(annotated_image, 
                         (text_x - padding, text_y - text_height - padding),
                         (text_x + text_width + padding, text_y + baseline + padding),
                         (0, 0, 139), -1)  # 深红色背景
            
            # 绘制白色文本
            cv2.putText(annotated_image, text, (text_x, text_y), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # 转换为JPEG二进制流
        _, buffer = cv2.imencode('.jpg', annotated_image)
        return buffer.tobytes(), detections
    
    def predict_single_image(self, image_path: str) -> tuple:
        """预测单张图片文件"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        return self.predict_image(image)
    
    def predict_images_folder(self, folder_path: str) -> List[tuple]:
        """预测文件夹中的所有图片"""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        results = []
        folder = Path(folder_path)
        
        for image_file in folder.iterdir():
            if image_file.suffix.lower() in image_extensions:
                try:
                    jpeg_data, detections = self.predict_single_image(str(image_file))
                    results.append((image_file.name, jpeg_data, detections))
                    print(f"已处理: {image_file.name}")
                except Exception as e:
                    print(f"处理 {image_file.name} 时出错: {e}")
        
        return results
    
    def predict_video_stream(self, stream_url: str, max_frames: int = 100) -> List[tuple]:
        """预测网络视频流"""
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频流: {stream_url}")
        
        results = []
        frame_count = 0
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("视频流结束或读取失败")
                    break
                
                try:
                    jpeg_data, detections = self.predict_image(frame)
                    results.append((jpeg_data, detections))
                    frame_count += 1
                    print(f"已处理帧: {frame_count}")
                except Exception as e:
                    print(f"处理第 {frame_count} 帧时出错: {e}")
        
        finally:
            cap.release()
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO预测脚本')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径(.pt)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS阈值')
    
    # 输入源参数（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='单张图片路径')
    input_group.add_argument('--folder', type=str, help='图片文件夹路径')
    input_group.add_argument('--stream', type=str, help='网络视频流URL')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    parser.add_argument('--max-frames', type=int, default=100, help='视频流最大处理帧数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 初始化预测器
        predictor = YOLOPredictor(
            model_path=args.model,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
        
        if args.image:
            # 单张图片预测
            print(f"预测单张图片: {args.image}")
            jpeg_data, detections = predictor.predict_single_image(args.image)
            
            # 保存结果
            output_path = os.path.join(args.output, 'predicted_image.jpg')
            with open(output_path, 'wb') as f:
                f.write(jpeg_data)
            print(f"结果已保存到: {output_path}")
        
        elif args.folder:
            # 文件夹图片预测
            print(f"预测文件夹图片: {args.folder}")
            results = predictor.predict_images_folder(args.folder)
            
            # 保存结果
            for filename, jpeg_data, detections in results:
                output_path = os.path.join(args.output, f'predicted_{filename}')
                with open(output_path, 'wb') as f:
                    f.write(jpeg_data)
            print(f"共处理 {len(results)} 张图片，结果已保存到: {args.output}")
        
        elif args.stream:
            # 视频流预测
            print(f"预测视频流: {args.stream}")
            results = predictor.predict_video_stream(args.stream, args.max_frames)
            
            # 保存结果
            for i, (jpeg_data, detections) in enumerate(results):
                output_path = os.path.join(args.output, f'frame_{i:04d}.jpg')
                with open(output_path, 'wb') as f:
                    f.write(jpeg_data)
            print(f"共处理 {len(results)} 帧，结果已保存到: {args.output}")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()