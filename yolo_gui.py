#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO目标检测GUI界面
基于tkinter的图形用户界面，支持单张图片、批量图片和视频流取帧检测
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from PIL import Image, ImageTk
import io
import sys
from pathlib import Path

# 添加yolov5路径
sys.path.append(str(Path(__file__).parent / 'yolov5'))
from yolo_predict import YOLOPredictor

class YOLODetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO目标检测系统")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.predictor = None
        self.current_image = None
        self.processing = False
        
        # 图片源切换相关变量
        self.showing_original = True  # True表示显示原始图片，False表示显示检测结果
        self.original_image_path = None  # 当前原始图片路径
        self.detection_result_path = None  # 当前检测结果图片路径
        
        # 创建界面
        self.create_widgets()
        
        # 启动时自动扫描模型文件
        self.root.after(100, self.auto_scan_models)  # 延迟100ms执行，确保界面完全加载
        
    def create_widgets(self):
        """创建GUI组件"""
        # 创建主框架，padding="10"设置内边距为10像素
        main_frame = ttk.Frame(self.root, padding="10")
        # row=0第0行，column=0第0列，sticky=(tk.W, tk.E, tk.N, tk.S)四方向拉伸填充
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        # 根窗口第0列权重为1，使其可以水平拉伸
        self.root.columnconfigure(0, weight=1)
        # 根窗口第0行权重为1，使其可以垂直拉伸
        self.root.rowconfigure(0, weight=1)
        # 主框架第1列权重为1，使其可以水平拉伸
        main_frame.columnconfigure(1, weight=1)
        # 主框架第3行权重为1，使其可以垂直拉伸（图像显示区域）
        main_frame.rowconfigure(3, weight=1)
        
        # 创建模型配置区域
        self.create_model_config_frame(main_frame)
        
        # 创建控制按钮区域
        self.create_control_frame(main_frame)

        # 创建命令选项区域
        self.create_command_frame(main_frame)

        # 创建图像显示区域
        self.create_image_frame(main_frame)
        
        # 创建日志显示区域
        self.create_log_frame(main_frame)
        
        # 创建状态栏
        self.create_status_bar()
        
    def create_model_config_frame(self, parent):
        """创建模型配置区域"""
        # 创建带标题的框架，padding="10"设置内边距为10像素
        config_frame = ttk.LabelFrame(parent, text="模型配置", padding="10")
        # 网格布局：row=0第0行，column=0第0列，columnspan=1跨越1列，sticky=(tk.W, tk.E, tk.S, tk.N)水平拉伸填充，pady=(10, 10)上下边距10像素
        config_frame.grid(row=0, column=0, columnspan=1, sticky=(tk.W, tk.E, tk.S, tk.N), pady=(10, 10))
        
        # 模型路径标签
        # sticky=tk.W左对齐，padx=(0, 5)右边距5像素
        ttk.Label(config_frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_path_var = tk.StringVar()
        # width=50设置输入框宽度为50字符，sticky=(tk.W, tk.E)水平拉伸填充
        model_entry = ttk.Entry(config_frame, textvariable=self.model_path_var, width=50)
        model_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        # 浏览按钮，row=0第0行，column=2第2列
        ttk.Button(config_frame, text="浏览", command=self.browse_model).grid(row=0, column=2)
        
        # 置信度阈值标签
        # pady=(5, 0)上边距5像素，下边距0像素
        ttk.Label(config_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.conf_var = tk.DoubleVar(value=0.5)
        # from_=0.1最小值0.1，to=1.0最大值1.0，orient=tk.HORIZONTAL水平方向
        conf_scale = ttk.Scale(config_frame, from_=0.1, to=1.0, variable=self.conf_var, orient=tk.HORIZONTAL)
        conf_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        # 显示当前置信度值的标签
        self.conf_label = ttk.Label(config_frame, text="0.50")
        self.conf_label.grid(row=1, column=2, pady=(5, 0))
        # command绑定滑块值变化时的回调函数
        conf_scale.configure(command=self.update_conf_label)
        
        # IOU阈值标签
        ttk.Label(config_frame, text="IOU阈值:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.iou_var = tk.DoubleVar(value=0.5)
        # IOU阈值滑块，参数含义同置信度滑块
        iou_scale = ttk.Scale(config_frame, from_=0.1, to=1.0, variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        # 显示当前IOU值的标签
        self.iou_label = ttk.Label(config_frame, text="0.50")
        self.iou_label.grid(row=2, column=2, pady=(5, 0))
        # 绑定IOU滑块值变化的回调函数
        iou_scale.configure(command=self.update_iou_label)
        
        # 加载模型按钮，row=3第3行，column=1第1列，pady=(10, 0)上边距10像素
        ttk.Button(config_frame, text="加载模型", command=self.load_model).grid(row=3, column=1, pady=(10, 0))
        
        # 设置第1列（索引1）的权重为1，使其可以水平拉伸
        config_frame.columnconfigure(1, weight=1)
        
    def create_control_frame(self, parent):
        """创建控制按钮区域"""
        # 创建检测控制框架，padding="10"设置内边距为10像素
        control_frame = ttk.LabelFrame(parent, text="检测控制", padding="10")
        # row=1第1行，columnspan=1跨越1列，sticky=(tk.N, tk.S, tk.E, tk.W)垂直拉伸，pady=(0, 10)下边距10像素
        control_frame.grid(row=1, column=0, columnspan=1, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(10, 10))
        
        # 当前选择路径显示
        # row=1第1行，column=0第0列，sticky=tk.W左对齐，pady=(10, 0)上边距10像素
        ttk.Label(control_frame, text="当前路径:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.current_path_var = tk.StringVar(value="未选择文件")
        # 当前路径显示输入框，width=50设置宽度为50字符，state="readonly"只读状态
        current_path_entry = ttk.Entry(control_frame, textvariable=self.current_path_var, width=50, state="readonly")
        # columnspan=2跨越2列，sticky=(tk.W, tk.E)水平拉伸，padx=(5, 0)左边距5像素
        current_path_entry.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0), padx=(5, 0))
        
        # 视频流检测标签
        # row=2第2行，column=0第0列，sticky=tk.W左对齐，pady=(10, 0)上边距10像素
        ttk.Label(control_frame, text="视频流URL:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.stream_url_var = tk.StringVar()
        # width=30设置输入框宽度为30字符
        stream_entry = ttk.Entry(control_frame, textvariable=self.stream_url_var, width=50)
        self.stream_url_var.set("rtsp://admin:1qaz!QAZ@192.168.1.108:554")  # 设置默认值
        # columnspan=2跨越2列，sticky=(tk.W, tk.E)水平拉伸，padx=(5, 0)左边距5像素，右边距0像素
        stream_entry.grid(row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0), padx=(5, 0))

        # 检测帧数设置
        # row=3第3行，column=0第0列，sticky=tk.W左对齐，pady=(10, 0)上边距10像素
        ttk.Label(control_frame, text="检测帧数:").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        self.max_frames_var = tk.IntVar(value=100)
        # 帧数输入框，width=10设置宽度为10字符
        frames_entry = ttk.Entry(control_frame, textvariable=self.max_frames_var, width=10)
        frames_entry.grid(row=3, column=1, sticky=tk.W, pady=(10, 0), padx=(5, 5))
        # 帧数说明标签
        ttk.Label(control_frame, text="(1-1000帧)").grid(row=3, column=2, sticky=tk.W, pady=(10, 0))
        
        # 输出目录标签
        # row=4第4行，column=0第0列，sticky=tk.W左对齐，pady=(10, 0)上边距10像素
        ttk.Label(control_frame, text="输出目录:").grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.output_dir_var = tk.StringVar(value="./output")
        # 输出目录输入框，width=30设置宽度为30字符
        output_entry = ttk.Entry(control_frame, textvariable=self.output_dir_var, width=30)
        # sticky=(tk.W, tk.E)水平拉伸，padx=(5, 5)左右边距各5像素
        output_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=(10, 0), padx=(5, 5))
        # 浏览输出目录按钮，row=4第4行，column=2第2列
        ttk.Button(control_frame, text="浏览", command=self.browse_output_dir).grid(row=4, column=2, pady=(10, 0), padx=(0, 5))
        # 打开输出目录按钮，row=4第4行，column=3第3列
        ttk.Button(control_frame, text="打开目录", command=self.open_output_dir).grid(row=4, column=3, pady=(10, 0))
        
        # 设置第1列（索引1）的权重为1，使其可以水平拉伸
        control_frame.columnconfigure(1, weight=1)
        # 设置第2列（索引2）的权重为1，使其可以水平拉伸
        control_frame.columnconfigure(2, weight=1)

    def create_command_frame(self, parent):
        """创建命令区域"""
        command_frame = ttk.LabelFrame(parent, text="命令选项", padding="10")
        command_frame.grid(row=2, column=0, columnspan=1, sticky=(tk.W, tk.E), pady=(10, 10), padx=(10, 10))
        # 设置第0行（索引0）的权重为1，使其可以垂直拉伸
        command_frame.rowconfigure(0, weight=1)

        # 单张图片检测按钮
        # row=0第0行，column=0第0列，padx=(0, 5)右边距5像素
        ttk.Button(command_frame, text="选择单张图片", command=self.select_single_image).grid(row=0, column=0, sticky=(tk.W,tk.E), padx=(5, 5), pady=(5, 5))
        # row=0第0行，column=1第1列，padx=(0, 5)右边距5像素
        ttk.Button(command_frame, text="检测单张", command=self.detect_current_image).grid(row=0, column=1, sticky=(tk.W,tk.E), padx=(5, 5), pady=(5, 5))

        # 批量图片检测按钮
        # row=0第0行，column=2第2列，padx=(0, 5)右边距5像素
        ttk.Button(command_frame, text="选择文件夹", command=self.select_image_folder).grid(row=0, column=2, sticky=(tk.W,tk.E), padx=(5, 5), pady=(5, 5))
        # row=0第0行，column=3第3列，padx=(0, 5)右边距5像素
        ttk.Button(command_frame, text="批量检测", command=self.batch_detect).grid(row=0, column=3, sticky=(tk.W,tk.E), padx=(5, 5), pady=(5, 5))

        # 开始视频流检测按钮，row=2第2行，column=3第3列
        ttk.Button(command_frame, text="开始视频流检测", command=self.start_stream_detection).grid(row=0, column=4, sticky=(tk.W,tk.E), pady=(5, 5), padx=(5, 5))

        command_frame.columnconfigure(0, weight=1)
        command_frame.columnconfigure(1, weight=1)
        command_frame.columnconfigure(2, weight=1)
        command_frame.columnconfigure(3, weight=1)
        command_frame.columnconfigure(4, weight=1)

    def create_image_frame(self, parent):
        """创建图像显示区域"""
        # 图像显示区域框架
        image_frame = ttk.LabelFrame(parent, text="图像显示", padding="10")
        image_frame.grid(row=0, column=1, columnspan=3, rowspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 10), pady=(10, 10))
        
        # 创建缩放控制区域
        zoom_frame = ttk.Frame(image_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        # 缩放控制按钮
        ttk.Button(zoom_frame, text="放大 (+)", command=self.zoom_in, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="缩小 (-)", command=self.zoom_out, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="适应窗口", command=self.fit_to_window, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_frame, text="原始大小", command=self.actual_size, width=10).pack(side=tk.LEFT, padx=(0, 5))
        
        # 图片源切换按钮
        self.toggle_source_btn = ttk.Button(zoom_frame, text="显示检测结果", command=self.toggle_image_source, width=12)
        self.toggle_source_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        # 缩放比例显示
        self.zoom_var = tk.StringVar(value="100%")
        ttk.Label(zoom_frame, textvariable=self.zoom_var).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 创建图片选择区域（当选择文件夹时显示）
        self.image_selector_frame = ttk.Frame(image_frame)
        
        # 图片选择标签和下拉框
        ttk.Label(self.image_selector_frame, text="选择图片:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_list_var = tk.StringVar()
        self.image_combobox = ttk.Combobox(self.image_selector_frame, textvariable=self.image_list_var, 
                                          state="readonly", width=40)
        self.image_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.image_combobox.bind('<<ComboboxSelected>>', self.on_image_selected)
        
        # 导航按钮
        ttk.Button(self.image_selector_frame, text="上一张", command=self.prev_image, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.image_selector_frame, text="下一张", command=self.next_image, width=8).pack(side=tk.LEFT, padx=(0, 5))
        
        # 图片计数显示
        self.image_count_var = tk.StringVar(value="")
        ttk.Label(self.image_selector_frame, textvariable=self.image_count_var).pack(side=tk.RIGHT, padx=(10, 0))
        
        # 初始化图片列表相关变量
        self.current_image_list = []
        self.current_image_index = -1
        self.current_folder_path = None  # 当前选择的文件夹路径
        self.output_folder_path = None   # 检测结果输出文件夹路径
        
        # 创建带滚动条的Canvas
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Canvas和滚动条
        self.image_canvas = tk.Canvas(canvas_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # 布局Canvas和滚动条
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 绑定鼠标滚轮事件
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.image_canvas.bind("<Control-MouseWheel>", self.on_ctrl_mousewheel)
        
        # 绑定键盘事件（需要设置焦点）
        self.image_canvas.bind("<Left>", lambda e: self.prev_image())
        self.image_canvas.bind("<Right>", lambda e: self.next_image())
        self.image_canvas.bind("<Button-1>", lambda e: self.image_canvas.focus_set())  # 点击获取焦点
        self.image_canvas.focus_set()  # 初始设置焦点
        
        # 初始化图像相关变量
        self.original_image = None
        self.current_photo = None
        self.zoom_factor = 1.0
        self.canvas_image_id = None
        
        # 显示提示文本
        self.image_canvas.create_text(400, 300, text="请选择图片进行检测", font=("Arial", 16), fill="gray")
        
    def create_log_frame(self, parent):
        """创建日志显示区域"""
        # 日志显示区域框架
        # padding="5"设置内边距为5像素
        log_frame = ttk.LabelFrame(parent, text="运行日志", padding="10")
        # row=2第2行，column=0第0列，sticky四方向拉伸，padx=(5, 0)左边距5像素，pady=(5, 0)上边距5像素
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 10), pady=(10, 10))
        
        # 带滚动条的文本框，width=30宽度30字符，height=20高度20行
        self.log_text = scrolledtext.ScrolledText(log_frame, width=30, height=20)
        # fill=tk.BOTH水平和垂直填充，expand=True允许扩展
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self):
        """创建状态栏"""
        # 状态文本变量，初始值为"就绪"
        self.status_var = tk.StringVar(value="就绪")
        # 状态栏标签，relief=tk.SUNKEN凹陷效果，anchor=tk.W左对齐
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        # row=1第1行，column=0第0列，sticky=(tk.W, tk.E)水平拉伸
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def log_message(self, message):
        """添加日志消息"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _log_detection_results(self, detections, image_name):
        """在日志中显示检测结果"""
        if not detections:
            self.log_message(f"图片 {image_name}: 无识别框")
        else:
            self.log_message(f"图片 {image_name}: 发现 {len(detections)} 个目标")
            for i, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                cls = det['class']
                
                # 获取类别名称
                if self.predictor and self.predictor.class_names and cls < len(self.predictor.class_names):
                    class_name = self.predictor.class_names[cls]
                else:
                    class_name = f"类别{cls}"
                
                # 计算中心点坐标
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                self.log_message(f"  目标{i}: {class_name} (置信度: {conf:.2f}) "
                               f"位置: ({x1},{y1})-({x2},{y2}) 中心: ({center_x},{center_y})")
        
    def update_conf_label(self, value):
        """更新置信度标签"""
        self.conf_label.config(text=f"{float(value):.2f}")
        
    def update_iou_label(self, value):
        """更新IOU标签"""
        self.iou_label.config(text=f"{float(value):.2f}")
        
    def browse_model(self):
        """浏览模型文件"""
        filename = filedialog.askopenfilename(
            title="选择YOLO模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
            
    def browse_output_dir(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir_var.set(dirname)
    
    def auto_scan_models(self):
        """自动扫描工作目录下的.pt文件"""
        try:
            # 获取当前工作目录
            current_dir = os.getcwd()
            pt_files = []
            
            # 扫描当前目录及子目录中的.pt文件
            for root, dirs, files in os.walk(current_dir):
                for file in files:
                    if file.endswith('.pt'):
                        pt_files.append(os.path.join(root, file))
            
            if not pt_files:
                self.log_message("自动扫描: 未找到.pt模型文件")
                return
            
            if len(pt_files) == 1:
                # 只有一个文件，直接设置
                self.model_path_var.set(pt_files[0])
                self.log_message(f"自动扫描: 找到并设置模型文件 {os.path.basename(pt_files[0])}")
            else:
                # 多个文件，自动选择第一个，并在日志中显示所有找到的文件
                self.model_path_var.set(pt_files[0])
                self.log_message(f"自动扫描: 找到 {len(pt_files)} 个.pt文件，已自动选择 {os.path.basename(pt_files[0])}")
                self.log_message(f"其他可用模型: {', '.join([os.path.basename(f) for f in pt_files[1:]])}")
                
        except Exception as e:
            error_msg = f"自动扫描模型文件时出错: {str(e)}"
            self.log_message(error_msg)
    
    def open_output_dir(self):
        """打开输出目录"""
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showwarning("警告", "输出目录未设置")
            return
        
        # 如果目录不存在，先创建它
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log_message(f"创建输出目录: {output_dir}")
            except Exception as e:
                messagebox.showerror("错误", f"无法创建输出目录: {str(e)}")
                return
        
        # 使用系统默认程序打开目录
        try:
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
            
            self.log_message(f"已打开输出目录: {output_dir}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {str(e)}")
            
    def load_model(self):
        """加载YOLO模型"""
        model_path = self.model_path_var.get()
        if not model_path:
            messagebox.showerror("错误", "请先选择模型文件")
            return
            
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "模型文件不存在")
            return
            
        try:
            self.status_var.set("正在加载模型...")
            self.log_message(f"开始加载模型: {model_path}")
            
            self.predictor = YOLOPredictor(
                model_path=model_path,
                conf_thres=self.conf_var.get(),
                iou_thres=self.iou_var.get()
            )
            
            self.log_message("模型加载成功")
            self.status_var.set("模型已加载")
            messagebox.showinfo("成功", "模型加载成功")
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.log_message(error_msg)
            self.status_var.set("模型加载失败")
            messagebox.showerror("错误", error_msg)
            
    def select_single_image(self):
        """选择单张图片"""
        filename = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("所有文件", "*.*")]
        )
        if filename:
            self.current_image = filename
            self.current_path_var.set(filename)  # 更新路径显示
            
            # 重置图片源状态
            self.showing_original = True
            self.original_image_path = filename
            self.detection_result_path = None
            self.toggle_source_btn.config(text="显示检测结果")
            
            self.display_image(filename)
            self.log_message(f"已选择图片: {os.path.basename(filename)}")
            
            # 隐藏文件夹图片选择控件
            self.image_selector_frame.pack_forget()
            self.current_image_list = []
            self.current_image_index = -1
            
    def display_image(self, image_path):
        """显示图片"""
        try:
            # 加载原始图片
            self.original_image = Image.open(image_path)
            self.zoom_factor = 1.0
            
            # 更新当前图片路径（仅在显示原始图片时更新）
            if self.showing_original:
                self.original_image_path = image_path
            
            # 清除Canvas内容
            self.image_canvas.delete("all")
            
            # 显示图片
            self._update_image_display()
            
        except Exception as e:
            self.log_message(f"显示图片失败: {str(e)}")
            
    def _update_image_display(self):
        """更新图片显示"""
        if self.original_image is None:
            return
            
        try:
            # 计算缩放后的尺寸
            width = int(self.original_image.width * self.zoom_factor)
            height = int(self.original_image.height * self.zoom_factor)
            
            # 缩放图片
            if self.zoom_factor == 1.0:
                display_image = self.original_image
            else:
                display_image = self.original_image.resize((width, height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            self.current_photo = ImageTk.PhotoImage(display_image)
            
            # 清除旧图片
            if self.canvas_image_id:
                self.image_canvas.delete(self.canvas_image_id)
            
            # 在Canvas中心显示图片
            self.canvas_image_id = self.image_canvas.create_image(
                width // 2, height // 2, 
                image=self.current_photo, 
                anchor=tk.CENTER
            )
            
            # 更新滚动区域
            self.image_canvas.configure(scrollregion=(0, 0, width, height))
            
            # 更新缩放比例显示
            self.zoom_var.set(f"{int(self.zoom_factor * 100)}%")
            
        except Exception as e:
            self.log_message(f"更新图片显示失败: {str(e)}")
    
    def zoom_in(self):
        """放大图片"""
        if self.original_image is None:
            return
        self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)  # 最大放大5倍
        self._update_image_display()
        
    def zoom_out(self):
        """缩小图片"""
        if self.original_image is None:
            return
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)  # 最小缩小到10%
        self._update_image_display()
        
    def fit_to_window(self):
        """适应窗口大小"""
        if self.original_image is None:
            return
            
        # 获取Canvas的实际显示区域大小
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas还未完全初始化，使用默认值
            canvas_width = 800
            canvas_height = 600
            
        # 计算适应窗口的缩放比例
        width_ratio = canvas_width / self.original_image.width
        height_ratio = canvas_height / self.original_image.height
        self.zoom_factor = min(width_ratio, height_ratio, 1.0)  # 不超过原始大小
        
        self._update_image_display()
        
    def actual_size(self):
        """显示原始大小"""
        if self.original_image is None:
            return
        self.zoom_factor = 1.0
        self._update_image_display()
        
    def on_mousewheel(self, event):
        """处理鼠标滚轮事件（滚动）"""
        # 垂直滚动
        self.image_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def on_ctrl_mousewheel(self, event):
        """处理Ctrl+鼠标滚轮事件（缩放）"""
        if self.original_image is None:
            return
            
        # 获取鼠标在Canvas中的位置
        x = self.image_canvas.canvasx(event.x)
        y = self.image_canvas.canvasy(event.y)
        
        # 根据滚轮方向缩放
        if event.delta > 0:
            scale = 1.1
        else:
            scale = 0.9
            
        old_zoom = self.zoom_factor
        self.zoom_factor = max(min(self.zoom_factor * scale, 5.0), 0.1)
        
        if self.zoom_factor != old_zoom:
            # 计算缩放后需要调整的滚动位置，使鼠标位置保持不变
            scale_ratio = self.zoom_factor / old_zoom
            
            self._update_image_display()
            
            # 调整滚动位置
            new_x = x * scale_ratio
            new_y = y * scale_ratio
            
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            scroll_x = (new_x - event.x) / (self.original_image.width * self.zoom_factor)
            scroll_y = (new_y - event.y) / (self.original_image.height * self.zoom_factor)
            
            self.image_canvas.xview_moveto(max(0, min(1, scroll_x)))
            self.image_canvas.yview_moveto(max(0, min(1, scroll_y)))
    
    def toggle_image_source(self):
        """切换图片显示源（原始图片 <-> 检测结果）"""
        if not self.original_image_path:
            messagebox.showwarning("警告", "请先选择图片！")
            return
        
        if self.showing_original:
            # 当前显示原始图片，切换到检测结果
            # 在文件夹模式下，需要动态计算检测结果路径
            if self.current_image_list and self.current_image_index >= 0:
                # 文件夹模式：基于当前选择的图片计算检测结果路径
                current_image_path = self.current_image_list[self.current_image_index]
                output_filename = f"detected_{os.path.basename(current_image_path)}"
                potential_result_path = os.path.join(self.output_dir_var.get(), output_filename)
                if os.path.exists(potential_result_path):
                    self.detection_result_path = potential_result_path
                else:
                    self.detection_result_path = None
            
            if self.detection_result_path and os.path.exists(self.detection_result_path):
                self.showing_original = False
                self.display_image(self.detection_result_path)
                self.toggle_source_btn.config(text="显示原始图片")
                self.log_message("切换到检测结果显示")
            else:
                messagebox.showinfo("提示", "检测结果不存在，请先进行检测！")
        else:
            # 当前显示检测结果，切换到原始图片
            self.showing_original = True
            self.display_image(self.original_image_path)
            self.toggle_source_btn.config(text="显示检测结果")
            self.log_message("切换到原始图片显示")
             
    def detect_current_image(self):
        """检测当前图片"""
        if not self.predictor:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        if not self.current_image:
            messagebox.showerror("错误", "请先选择图片")
            return
            
        if self.processing:
            messagebox.showwarning("警告", "正在处理中，请稍候")
            return
            
        # 在新线程中执行检测
        threading.Thread(target=self._detect_single_image, daemon=True).start()
        
    def _detect_single_image(self):
        """在后台线程中检测单张图片"""
        try:
            self.processing = True
            self.status_var.set("正在检测...")
            self.log_message(f"开始检测图片: {os.path.basename(self.current_image)}")
            
            # 执行检测
            jpeg_data, detections = self.predictor.predict_single_image(self.current_image)
            
            # 保存结果
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(self.current_image)}")
            
            with open(output_path, 'wb') as f:
                f.write(jpeg_data)
                
            # 更新检测结果路径
            self.detection_result_path = output_path
            
            # 切换到显示检测结果
            self.showing_original = False
            self.root.after(0, lambda: self.display_image(output_path))
            self.root.after(0, lambda: self.toggle_source_btn.config(text="显示原始图片"))
            
            # 在日志中显示检测结果
            self._log_detection_results(detections, os.path.basename(self.current_image))
            
            self.log_message(f"检测完成，结果保存到: {output_path}")
            self.status_var.set("检测完成")
            
        except Exception as e:
            error_msg = f"检测失败: {str(e)}"
            self.log_message(error_msg)
            self.status_var.set("检测失败")
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
        finally:
            self.processing = False
            
    def select_image_folder(self):
        """选择图片文件夹"""
        dirname = filedialog.askdirectory(title="选择包含图片的文件夹")
        if dirname:
            self.current_folder = dirname
            self.current_folder_path = dirname
            self.current_path_var.set(dirname)  # 更新路径显示
            self.log_message(f"已选择文件夹: {dirname}")
            
            # 重置图片源状态
            self.showing_original = True
            self.detection_result_path = None
            self.toggle_source_btn.config(text="显示检测结果")
            
            # 扫描文件夹中的图片文件
            self._scan_folder_images(dirname)
            
    def _scan_folder_images(self, folder_path):
        """扫描文件夹中的图片文件"""
        try:
            # 支持的图片格式
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
            
            # 扫描文件夹
            image_files = []
            for file in os.listdir(folder_path):
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_files.append(file)
            
            # 按文件名排序
            image_files.sort()
            
            # 更新图片列表
            self.current_image_list = [os.path.join(folder_path, f) for f in image_files]
            
            if self.current_image_list:
                # 显示图片选择控件
                self.image_selector_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 5))
                
                # 更新下拉框选项
                self.image_combobox['values'] = image_files
                
                # 选择第一张图片
                self.current_image_index = 0
                self.image_combobox.current(0)
                self._update_image_selection()
                
                self.log_message(f"找到 {len(self.current_image_list)} 张图片")
            else:
                # 隐藏图片选择控件
                self.image_selector_frame.pack_forget()
                self.current_image_list = []
                self.current_image_index = -1
                
                # 清除图片显示
                self.image_canvas.delete("all")
                self.image_canvas.create_text(400, 300, text="文件夹中没有找到图片文件", 
                                            font=("Arial", 16), fill="gray")
                
                self.log_message("文件夹中没有找到图片文件")
                
        except Exception as e:
             self.log_message(f"扫描文件夹失败: {str(e)}")
             messagebox.showerror("错误", f"扫描文件夹失败: {str(e)}")
    
    def on_image_selected(self, event=None):
        """处理图片选择事件"""
        if not self.current_image_list:
            return
            
        selected_index = self.image_combobox.current()
        if 0 <= selected_index < len(self.current_image_list):
            self.current_image_index = selected_index
            self._update_image_selection()
            
    def _update_image_selection(self):
        """更新图片选择显示"""
        if not self.current_image_list or self.current_image_index < 0:
            return
            
        try:
            # 获取当前选择的图片路径
            current_image_path = self.current_image_list[self.current_image_index]
            
            # 更新当前图片变量
            self.current_image = current_image_path
            
            # 重置图片源状态
            self.showing_original = True
            self.original_image_path = current_image_path
            
            # 更新检测结果路径（基于输出文件夹）
            if self.output_dir_var.get():
                output_filename = f"detected_{os.path.basename(current_image_path)}"
                potential_result_path = os.path.join(self.output_dir_var.get(), output_filename)
                if os.path.exists(potential_result_path):
                    self.detection_result_path = potential_result_path
                else:
                    self.detection_result_path = None
            else:
                self.detection_result_path = None
            
            self.toggle_source_btn.config(text="显示检测结果")
            
            # 显示图片
            self.display_image(current_image_path)
            
            # 更新计数显示
            total_count = len(self.current_image_list)
            current_num = self.current_image_index + 1
            self.image_count_var.set(f"{current_num}/{total_count}")
            
            # 更新路径显示为当前图片路径
            self.current_path_var.set(current_image_path)
            
            self.log_message(f"显示图片: {os.path.basename(current_image_path)} ({current_num}/{total_count})")
            
        except Exception as e:
            self.log_message(f"显示图片失败: {str(e)}")
            
    def prev_image(self):
        """显示上一张图片"""
        if not self.current_image_list:
            return
            
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_combobox.current(self.current_image_index)
            self._update_image_selection()
        else:
            # 循环到最后一张
            self.current_image_index = len(self.current_image_list) - 1
            self.image_combobox.current(self.current_image_index)
            self._update_image_selection()
            
    def next_image(self):
        """显示下一张图片"""
        if not self.current_image_list:
            return
            
        if self.current_image_index < len(self.current_image_list) - 1:
            self.current_image_index += 1
            self.image_combobox.current(self.current_image_index)
            self._update_image_selection()
        else:
            # 循环到第一张
            self.current_image_index = 0
            self.image_combobox.current(self.current_image_index)
            self._update_image_selection()
             
    def batch_detect(self):
        """批量检测图片"""
        if not self.predictor:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        if not hasattr(self, 'current_folder'):
            messagebox.showerror("错误", "请先选择图片文件夹")
            return
            
        if self.processing:
            messagebox.showwarning("警告", "正在处理中，请稍候")
            return
            
        # 在新线程中执行批量检测
        threading.Thread(target=self._batch_detect, daemon=True).start()
        
    def _batch_detect(self):
        """在后台线程中批量检测图片"""
        try:
            self.processing = True
            self.status_var.set("正在批量检测...")
            self.log_message(f"开始批量检测文件夹: {self.current_folder}")
            
            # 执行批量检测
            results = self.predictor.predict_images_folder(self.current_folder)
            
            # 保存结果
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            
            for filename, jpeg_data, detections in results:
                output_path = os.path.join(output_dir, f"detected_{filename}")
                with open(output_path, 'wb') as f:
                    f.write(jpeg_data)
                # 在日志中显示每张图片的检测结果
                self._log_detection_results(detections, filename)
                    
            self.log_message(f"批量检测完成，共处理 {len(results)} 张图片")
            self.status_var.set("批量检测完成")
            
            self.root.after(0, lambda: messagebox.showinfo("完成", f"批量检测完成，共处理 {len(results)} 张图片"))
            
        except Exception as e:
            error_msg = f"批量检测失败: {str(e)}"
            self.log_message(error_msg)
            self.status_var.set("批量检测失败")
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
        finally:
            self.processing = False
            
    def start_stream_detection(self):
        """开始视频流检测"""
        if not self.predictor:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        stream_url = self.stream_url_var.get()
        if not stream_url:
            messagebox.showerror("错误", "请输入视频流URL")
            return
        
        # 验证帧数设置
        try:
            max_frames = self.max_frames_var.get()
            if max_frames < 1 or max_frames > 1000:
                messagebox.showerror("错误", "检测帧数必须在1-1000之间")
                return
        except tk.TclError:
            messagebox.showerror("错误", "请输入有效的帧数")
            return
            
        if self.processing:
            messagebox.showwarning("警告", "正在处理中，请稍候")
            return
            
        # 在新线程中执行视频流检测
        threading.Thread(target=self._stream_detect, args=(stream_url,), daemon=True).start()
        
    def _stream_detect(self, stream_url):
        """在后台线程中检测视频流"""
        try:
            self.processing = True
            self.status_var.set("正在检测视频流...")
            self.log_message(f"开始检测视频流: {stream_url}")
            
            # 执行视频流检测
            max_frames = self.max_frames_var.get()
            results = self.predictor.predict_video_stream(stream_url, max_frames=max_frames)
            
            # 保存结果
            output_dir = self.output_dir_var.get()
            os.makedirs(output_dir, exist_ok=True)
            
            last_frame_path = None
            for i, (jpeg_data, detections) in enumerate(results):
                output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                with open(output_path, 'wb') as f:
                    f.write(jpeg_data)
                last_frame_path = output_path  # 记录最后一帧的路径
                # 在日志中显示每帧的检测结果
                self._log_detection_results(detections, f"帧{i:04d}")
            
            self.log_message(f"视频流检测完成，共处理 {len(results)} 帧")
            self.status_var.set("视频流检测完成")
            
            # 在主线程中显示最后一帧图片
            if last_frame_path and os.path.exists(last_frame_path):
                self.root.after(0, lambda: self._display_stream_result(last_frame_path, len(results)))
            else:
                self.root.after(0, lambda: messagebox.showinfo("完成", f"视频流检测完成，共处理 {len(results)} 帧"))
            
        except Exception as e:
            error_msg = f"视频流检测失败: {str(e)}"
            self.log_message(error_msg)
            self.status_var.set("视频流检测失败")
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            
        finally:
            self.processing = False

    def _display_stream_result(self, image_path, frame_count):
        """显示视频流检测结果"""
        try:
            # 更新当前路径显示
            self.current_path_var.set(image_path)
            
            # 设置图片路径变量
            self.original_image_path = None  # 视频流检测没有原始图片
            self.detection_result_path = image_path
            self.showing_original = False  # 显示检测结果
            
            # 显示图片
            self.display_image(image_path)
            
            # 显示完成消息
            messagebox.showinfo("完成", f"视频流检测完成，共处理 {frame_count} 帧\n最后一帧已显示在图像区域")
            
        except Exception as e:
            self.log_message(f"显示检测结果失败: {str(e)}")
            messagebox.showinfo("完成", f"视频流检测完成，共处理 {frame_count} 帧")

def main():
    """主函数"""
    root = tk.Tk()
    root.iconbitmap("icon.ico")
    app = YOLODetectionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()