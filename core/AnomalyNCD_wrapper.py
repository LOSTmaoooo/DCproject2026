# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：AnomalyNCD 算法封装器 (AnomalyNCD Wrapper)
# 文件名：anomalyncd_wrapper.py
# 功能：
#   1. 封装 AnomalyNCD (Novel Class Discovery) 算法。
#   2. 负责配置参数的构建、模型的初始化和运行。
#   3. 实现 `run` 方法，作为外部调用的统一接口，执行从二值化掩模到聚类发现的全流程。
# =================================================================================================

import os
import sys
import argparse
import torch

# -------------------------------------------------------------------------------------------------
# 模块清理与动态导入
# -------------------------------------------------------------------------------------------------
# 清理 sys.modules，防止与 MuSc 或其他库中的同名模块 ('utils', 'models' 等) 发生冲突。
for mod in ['utils', 'models', 'datasets']:
    if mod in sys.modules:
        del sys.modules[mod]

# 动态定位 AnomalyNCD 库的路径并加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCD_PATH = os.path.join(PROJECT_ROOT, 'libs', 'AnomalyNCD')
if NCD_PATH not in sys.path:
    sys.path.insert(0, NCD_PATH)

# 从 AnomalyNCD 库导入核心类
# AnomalyNCD: 算法主类
# utils.general_utils: 通用工具函数 (如 YAML 加载)
from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml as ncd_load_yaml

class ArgsStruct:
    """
    类：参数结构体 (Arguments Structure)
    
    功能：
        这是一个简单的辅助类，用于将字典 (Dictionary) 转换为对象 (Object)。
        目的：AnomalyNCD 的源代码通常使用 argparse 解析命令行参数 (args.param)，
        为了兼容其代码风格，我们将配置字典转换为这个对象的属性。
    """
    def __init__(self, **entries):
        # __dict__.update 将字典的键值对直接注入到对象的属性中
        self.__dict__.update(entries)

class AnomalyNCDWrapper:
    """
    类：AnomalyNCD 封装器
    
    功能：
        负责加载配置、准备运行目录、构建参数对象，并实例化执行 AnomalyNCD 模型。
    """
    
    def __init__(self, config_path=None):
        """
        初始化封装器。
        
        Args:
            config_path: AnomalyNCD 的 YAML 配置文件路径。如果为 None，则使用默认路径。
        """
        if config_path is None:
            # 默认指向 libs/AnomalyNCD/configs/AnomalyNCD.yaml
            config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
            
        self.config_path = config_path
        # 读取 YAML 配置并在内存中持有，以便后续构建参数
        self.cfg = ncd_load_yaml(config_path)
        
    def run(self, dataset_path, anomaly_map_path, base_data_path, output_dir):
        """
        方法：执行 AnomalyNCD 流程 (Run Process)
        
        功能：
            1. 准备输出目录结构 (experiment, binary_masks, crops)。
            2. 构造完整的参数字典 (合并传入路径和 YAML 配置)。
            3. 将参数转换为 ArgsStruct 对象。
            4. 实例化 AnomalyNCD 模型并调用其入口方法。
            
        Args:
            dataset_path: 包含待分析图像的目录 (必须符合 MTD 格式结构)。
            anomaly_map_path: 包含对应的异常热力图 (PNG格式) 的目录。
            base_data_path: 正常样本参考数据的目录。
            output_dir: 结果输出的根目录。
            
        Returns:
            bool: 成功返回 True，失败返回 False。
        """
        print(f"[AnomalyNCDWrapper] Initializing with dataset: {dataset_path}")
        
        # --- 1. 准备输出目录 ---
        # binary_masks: 存放二值化后的异常掩模
        binary_data_path = os.path.join(output_dir, 'binary_masks')
        # crops: 存放根据掩模裁剪出的异常区域 (Region of Interest)
        crop_data_path = os.path.join(output_dir, 'crops')
        # exp_root: 存放实验日志、Checkpoints 等
        exp_root = os.path.join(output_dir, 'experiment')
        
        # 递归创建目录
        os.makedirs(binary_data_path, exist_ok=True)
        os.makedirs(crop_data_path, exist_ok=True)
        os.makedirs(exp_root, exist_ok=True)
        
        # --- 2. 构建参数字典 (Argument Dictionary) ---
        # 这里的键名 (Key) 必须与 AnomalyNCD 内部 argparse 定义的参数名完全一致
        args_dict = {
            # --- 路径相关参数 ---
            'dataset_path': dataset_path,         # 输入图片根路径
            'anomaly_map_path': anomaly_map_path, # 输入热力图根路径
            'binary_data_path': binary_data_path, # 输出：二值掩模路径
            'crop_data_path': crop_data_path,     # 输出：裁剪图像路径
            'base_data_path': base_data_path,     # 参考：正常样本路径
            
            'dataset': 'custom',     # 数据集类型，指定为 'custom'
            'category': 'unknown',   # 类别名称，通常设为 'unknown'
            
            'config': self.config_path,  # 配置文件路径
            'runner_name': 'AnomalyNCD_Runner', # 运行器名称标识
            'only_test': None,           # 仅测试模式标志
            'checkpoint_path': None,     # 预训练模型路径
            
            # --- 二值化参数 (Binarization) ---
            # 从 YAML 配置中读取
            'sample_rate': self.cfg['binarization']['sample_rate'],       # 采样率
            'min_interval_len': self.cfg['binarization']['min_interval_len'], # 最小连通域长度
            'erode': self.cfg['binarization']['erode'],                   # 腐蚀操作参数
            
            # --- 模型架构参数 (Model Architecture) ---
            'grad_from_block': self.cfg['models']['grad_from_block'],     # 微调起始块索引
            'pretrained_backbone': self.cfg['models']['pretrained_backbone'], # 是否加载预训练骨干网络
            'mask_layers': self.cfg['models']['mask_layers'],             # 掩模层配置
            'n_views': self.cfg['models']['n_views'],                     # 对比学习的视图数量
            'n_head': self.cfg['models']['n_head'],                       # 注意力头数
            
            # --- 训练超参数 (Training Hyperparams) ---
            'batch_size': self.cfg['training']['batch_size'],    # 批次大小
            'num_workers': self.cfg['training']['num_workers'],  # 数据加载线程数
            'lr': self.cfg['training']['lr'],                    # 学习率
            'gamma': self.cfg['training']['gamma'],              # 学习率衰减系数
            'momentum': self.cfg['training']['momentum'],        # 动量
            'weight_decay': self.cfg['training']['weight_decay'],# 权重衰减 (L2 正则)
            'epochs': self.cfg['training']['epochs'],            # 训练轮数
            
            # --- 损失函数参数 (Loss Function) ---
            'sup_weight': self.cfg['loss']['sup_weight'],        # 监督损失权重
            'memax_weight': self.cfg['loss']['memax_weight'],    # 互信息最大化权重
            'anomaly_thred': self.cfg['loss']['anomaly_thred'],  # 异常阈值
            'teacher_temp': self.cfg['loss']['teacher_temp'],    # 教师网络温度系数
            'warmup_teacher_temp': self.cfg['loss']['warmup_teacher_temp'], # 预热温度
            'warmup_teacher_temp_epochs': self.cfg['loss']['warmup_teacher_temp_epochs'], # 预热轮数
            'repeat_times': self.cfg['loss']['repeat_times'],    # 重复次数
            
            # --- 实验与日志 (Logging) ---
            'seed': self.cfg['experiment']['seed'],              # 随机种子
            'print_freq': self.cfg['experiment']['print_freq'],  # 打印频率
            'table_root': os.path.join(output_dir, 'tables'),    # 表格结果输出路径
            'exp_name': 'discovery_run',                         # 实验名称
            'exp_root': exp_root                                 # 实验根目录
        }
        
        # --- 3. 转换为对象 ---
        args = ArgsStruct(**args_dict)
        
        # --- 4. 执行主要逻辑 ---
        print("[AnomalyNCDWrapper] Starting AnomalyNCD process...")
        try:
            # 实例化 AnomalyNCD 主类
            model = AnomalyNCD(args)
            
            # 检查并调用入口方法
            # 不同的 AnomalyNCD 版本可能入口方法名不同 (main vs train_init vs run)
            # 这里通过反射机制 (hasattr) 做一个兼容性处理
            if hasattr(model, 'main'):
                model.main()
            else:
                 # 回退方案
                 print("[AnomalyNCDWrapper] Warning: 'main' method not found. Calling train_init.")
                 model.train_init()
                 
        except Exception as e:
            # 捕获异常并记录
            print(f"[AnomalyNCDWrapper] Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

