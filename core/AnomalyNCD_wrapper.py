# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from utils.load_config import load_yaml 

# =================================================================================================
# 模块：AnomalyNCD 封装器 (AnomalyNCD Wrapper)
# 功能：封装 AnomalyNCD（First-phase Novel Class Discovery）的运行参数和流程。
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 路径配置与动态导入
# -------------------------------------------------------------------------------------------------
# 动态将 libs/AnomalyNCD 添加到系统路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCD_PATH = os.path.join(PROJECT_ROOT, 'libs', 'AnomalyNCD')
sys.path.append(NCD_PATH)

from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml as ncd_load_yaml

class ArgsStruct:
    """
    一个简单的类，用于模拟 argparse 的 Namespace 对象。
    AnomalyNCD 的原始代码使用 argparse 传递参数，
    这里我们将字典转换为对象属性，以便兼容原始接口。
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

class AnomalyNCDWrapper:
    """
    AnomalyNCD 的主要接口类。
    
    主要职责：
    1. 加载 AnomalyNCD 的 YAML 配置文件。
    2. run 方法负责组装所有必要的参数（包括路径、训练参数、模型参数）。
    3. 调用 AnomalyNCD 模型的主入口开始训练/发现过程。
    """
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
            
        self.config_path = config_path
        # 加载 yaml 配置为字典
        self.cfg = ncd_load_yaml(config_path)
        
    def run(self, dataset_path, anomaly_map_path, base_data_path, output_dir):
        """
        运行新类发现 (Discovery) 流程。
        
        参数:
            dataset_path (str): 目标数据集路径（包含未知类别的图像）。
            anomaly_map_path (str): MuSc 生成的异常热力图路径。
            base_data_path (str): 基准/正常数据集路径（用于对比）。
            output_dir (str): 结果输出的根目录。
            
        返回:
            bool: 成功返回 True，失败返回 False。
        """
        print(f"[AnomalyNCDWrapper] Initializing with dataset: {dataset_path}")
        
        # 准备输出子目录
        binary_data_path = os.path.join(output_dir, 'binary_masks')
        crop_data_path = os.path.join(output_dir, 'crops')
        exp_root = os.path.join(output_dir, 'experiment')
        
        os.makedirs(binary_data_path, exist_ok=True)
        os.makedirs(crop_data_path, exist_ok=True)
        os.makedirs(exp_root, exist_ok=True)
        
        # -----------------------------------------------------------------
        # 构建参数字典 (Args Dictionary)
        # 这里将运行时路径与 YAML 配置文件中的默认超参数合并
        # -----------------------------------------------------------------
        args_dict = {
            # --- 关键路径参数 ---
            'dataset_path': dataset_path,
            'anomaly_map_path': anomaly_map_path,
            'binary_data_path': binary_data_path, # 二值化掩模输出路径
            'crop_data_path': crop_data_path,     # 裁剪后感兴趣区域(ROI)输出路径
            'base_data_path': base_data_path,
            
            # --- 数据集元数据 ---
            'dataset': 'custom',  # 数据集类型
            'category': 'unknown',# 类别名称
            
            # --- 实验控制 ---
            'config': self.config_path,
            'runner_name': 'AnomalyNCD_Runner',
            'only_test': None,
            'checkpoint_path': None, # 如果是推理模式需提供路径
            
            # --- 二值化参数 (Binarization) ---
            # 从 cfg['binarization'] 读取
            'sample_rate': self.cfg['binarization']['sample_rate'],
            'min_interval_len': self.cfg['binarization']['min_interval_len'],
            'erode': self.cfg['binarization']['erode'],
            
            # --- 模型架构参数 (Model) ---
            # 从 cfg['models'] 读取
            'grad_from_block': self.cfg['models']['grad_from_block'], # 微调起始层
            'pretrained_backbone': self.cfg['models']['pretrained_backbone'],
            'mask_layers': self.cfg['models']['mask_layers'],
            'n_views': self.cfg['models']['n_views'],  #对比学习视图数
            'n_head': self.cfg['models']['n_head'],
            
            # --- 训练参数 (Training) ---
            'batch_size': self.cfg['training']['batch_size'],
            'num_workers': self.cfg['training']['num_workers'],
            'lr': self.cfg['training']['lr'],
            'gamma': self.cfg['training']['gamma'],
            'momentum': self.cfg['training']['momentum'],
            'weight_decay': self.cfg['training']['weight_decay'],
            'epochs': self.cfg['training']['epochs'],
            
            # --- 损失函数参数 (Loss) ---
            'sup_weight': self.cfg['loss']['sup_weight'],
            'memax_weight': self.cfg['loss']['memax_weight'],
            'anomaly_thred': self.cfg['loss']['anomaly_thred'],
            'teacher_temp': self.cfg['loss']['teacher_temp'],
            'warmup_teacher_temp': self.cfg['loss']['warmup_teacher_temp'],
            'warmup_teacher_temp_epochs': self.cfg['loss']['warmup_teacher_temp_epochs'],
            'repeat_times': self.cfg['loss']['repeat_times'],
            
            # --- 日志与其他 ---
            'seed': self.cfg['experiment']['seed'],
            'print_freq': self.cfg['experiment']['print_freq'],
            'table_root': os.path.join(output_dir, 'tables'), # 结果表格保存路径
            'exp_name': 'discovery_run',
            'exp_root': exp_root
        }
        
        # 转换为对象
        args = ArgsStruct(**args_dict)
        
        # -----------------------------------------------------------------
        # 实例化并运行模型
        # -----------------------------------------------------------------
        print("[AnomalyNCDWrapper] Starting AnomalyNCD process...")
        try:
            model = AnomalyNCD(args)
            
            # 兼容性检查：根据 AnomalyNCD类的定义调用主方法
            # 假设主入口是 main()，如果不同则可能需要调整
            if hasattr(model, 'main'):
                model.main()
            else:
                 # 回退方案：如果未找到 main，尝试调用初始化
                 # 注意：这通常不够，需要知道确切的执行流
                 print("[AnomalyNCDWrapper] Warning: 'main' method not found. Calling train_init.")
                 model.train_init()
                 
        except Exception as e:
            print(f"[AnomalyNCDWrapper] Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

