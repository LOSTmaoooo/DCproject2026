# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：核心执行引擎 (Core Execution Engine)
# 文件名：engine.py
# 功能：
#   1. 作为系统的中枢控制器，协调各个子模块（MuSc, DataBridge, AnomalyNCD）的协同工作。
#   2. 定义并执行批量处理流水线 (BatchPipeline)，管理数据流的传递。
#   3. 处理模块间的路径映射和资源初始化。
# =================================================================================================

import os
import sys
import shutil

# -------------------------------------------------------------------------------------------------
# 全局路径配置 (Global Path Configuration)
# -------------------------------------------------------------------------------------------------
# 动态获取项目根目录，确保代码在不同环境下都能正确找到资源。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 获取当前文件的上上级目录

# 定义关键子目录路径
LIBS_PATH = os.path.join(PROJECT_ROOT, 'libs')        # 第三方库/算法库目录
MUSC_PATH = os.path.join(LIBS_PATH, 'MuSc')           # MuSc 算法目录
NCD_PATH = os.path.join(LIBS_PATH, 'AnomalyNCD')      # AnomalyNCD 算法目录

# 将 core 目录加入系统路径，使其他模块可以导入 core 包下的内容
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

class BatchPipeline:
    """
    类：批量处理流水线 (Batch Processing Pipeline)
    
    功能：
        封装了完整的异常检测和新类发现流程。
        实现了 "MuSc生成热力图 -> DataBridge格式转换 -> AnomalyNCD聚类分析" 的逻辑串联。
    """
    
    def __init__(self):
        """
        构造函数：初始化流水线状态。
        注意：Wrapper 类采用延迟初始化 (Lazy Loading) 策略，初始设为 None。
        好处是启动引擎时不需要立即加载显存占用大的模型，只有在真正运行时才加载。
        """
        self.musc_wrapper = None # MuSc 算法封装器实例
        self.ncd_wrapper = None  # AnomalyNCD 算法封装器实例
        
        #设置默认的结果输出根目录：data_store/results
        self.output_base = os.path.join(PROJECT_ROOT, 'data_store', 'results')
        # 确保输出目录存在，如果不存在则创建
        os.makedirs(self.output_base, exist_ok=True)
        
    def run(self, input_dir, output_dir=None):
        """
        方法：执行完整流水线 (Run Pipeline)
        
        参数:
            input_dir (str): 输入目录，包含待检测的原始图像。
            output_dir (str, optional): 结果保存路径。如果不指定，则自动生成带时间戳的目录。
            
        返回:
            bool: 流水线执行成功返回 True，失败返回 False。
        """
        import time
        # 生成格式化的时间戳字符串 (例如: 20231027_103000)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 如果未指定输出目录，则在 base 目录下创建一个新目录
        if output_dir is None:
            output_dir = os.path.join(self.output_base, f"run_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BatchPipeline] Pipeline started on {input_dir}")
        print(f"[BatchPipeline] Results will be saved to {output_dir}")
        
        # =========================================================================
        # 阶段 1: 运行 MuSc 生成异常热力图 (Anomaly Map Generation)
        # =========================================================================
        print("\n--- Step 1: Running MuSc (Anomaly Map Generation) ---")
        try:
            # 调用内部方法 run_musc
            # 输入：原始图片目录
            # 输出：生成的 .npy 热力图所在的目录路径
            maps_dir = self.run_musc(input_dir, output_dir)
            
            # 校验结果：如果路径为空，说明 MuSc 运行失败或未生成文件
            if not maps_dir:
                print("[BatchPipeline] MuSc failed or produced no output.")
                return False
        except Exception as e:
            # 异常处理：打印错误并打印堆栈跟踪，方便调试
            print(f"[BatchPipeline] Error in MuSc step: {e}")
            import traceback
            traceback.print_exc()
            return False

        # =========================================================================
        # 阶段 2: 运行 DataBridge 数据转换 (Format Conversion)
        # =========================================================================
        print("\n--- Step 1.5: Running DataBridge (Format Conversion) ---")
        try:
            # 定义中间数据的存储目录
            interim_dir = os.path.join(output_dir, 'interim_ncd_data')
            
            # 调用 run_bridge 进行转换
            # 作用：将 MuSc 的输出 (.npy) 转换为 AnomalyNCD 需要的目录结构和格式 (.png)
            # 返回值：一个字典，包含转换后的图像路径和热力图路径
            bridge_result = self.run_bridge(input_dir, maps_dir, interim_dir)
            
            if not bridge_result:
                print("[BatchPipeline] DataBridge failed.")
                return False
                
            # 提取转换后的路径，准备传给下一级
            dataset_images_path = bridge_result['images_path']
            dataset_maps_path   = bridge_result['anomaly_maps_path']
                
        except Exception as e:
            print(f"[BatchPipeline] Error in DataBridge step: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # =========================================================================
        # 阶段 3: 运行 AnomalyNCD 新类发现 (Novel Class Discovery)
        # =========================================================================
        print("\n--- Step 2: Running AnomalyNCD (Novel Class Discovery) ---")
        try: 
            # 指定正常样本的参考数据集路径 (Normal Reference Data)
            # AnomalyNCD 算法通常需要对比正常样本来区分异常类型。
            base_data_path = os.path.join(PROJECT_ROOT, 'data_store', 'raw_inputs', 'normal_ref') 
            
            # 防御性编程：如果参考目录不存在，创建一个空目录防止程序崩溃
            # 注意：在实际生产环境中，这里应该报错或提示用户提供正常样本
            if not os.path.exists(base_data_path):
                 print(f"[BatchPipeline] Warning: Base data path {base_data_path} not found. Creating empty.")
                 os.makedirs(base_data_path, exist_ok=True)
            
            # 调用 run_ncd 执行核心聚类逻辑
            success = self.run_ncd(dataset_images_path, dataset_maps_path, base_data_path, output_dir)
            
            if not success:
                print("[BatchPipeline] AnomalyNCD step failed.")
                return False
                
        except Exception as e:
            print(f"[BatchPipeline] Error in AnomalyNCD step: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n[BatchPipeline] Pipeline finished successfully. Results in {output_dir}")
        return True

    def run_musc(self, input_dir, output_root):
        """
        方法：封装 MuSc 算法的调用逻辑
        
        返回:
            str: 生成的异常图 (.npy 文件) 所在的具体目录路径，失败返回 None。
        """
        from core.musc_wrapper import MuScWrapper  # 局部导入，避免循环依赖
        
        # 配置文件路径
        config_path = os.path.join(MUSC_PATH, 'configs', 'musc.yaml')
        # 输出子目录
        maps_output_dir = os.path.join(output_root, 'anomaly_maps')
        
        # 懒加载：如果是第一次调用，则初始化 Wrapper 实例
        if self.musc_wrapper is None:
            if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
            # 初始化 MuSc 模型 (加载权重到 GPU/CPU)
            self.musc_wrapper = MuScWrapper(config_path)
            
        # 执行批量生成
        saved_paths = self.musc_wrapper.generate_anomaly_maps(input_dir, maps_output_dir)
        
        # 检查是否生成了文件
        if saved_paths and len(saved_paths) > 0:
            return maps_output_dir
        return None

    def run_bridge(self, input_images, input_maps, output_dir):
        """
        方法：封装 DataBridge 的调用逻辑
        
        参数:
            input_images: 原始图片目录
            input_maps: MuSc 生成的 .npy 热力图目录
            output_dir: 转换后数据的存放目录
        """
        from core.data_bridge import DataBridge # 局部导入
        
        bridge = DataBridge() 
        
        print(f"[Engine] Bridging data from {input_images} to {output_dir}")
        # 执行转换
        result = bridge.prepare_ncd_dataset(input_images, input_maps, output_dir)
        return result  

    def run_ncd(self, dataset_path, anomaly_map_path, base_path, output_root):
        """
        方法：封装 AnomalyNCD 算法的调用逻辑
        
        参数:
            dataset_path: 整理好的图片目录 (DataBridge 输出)
            anomaly_map_path: 整理好的热力图目录 (DataBridge 输出)
            base_path: 正常样本目录
            output_root: 结果根目录
        """
        from core.anomalyncd_wrapper import AnomalyNCDWrapper # 局部导入
        
        config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
        ncd_output_dir = os.path.join(output_root, 'ncd_results')
        
        # 懒加载初始化
        if self.ncd_wrapper is None:
             if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
             self.ncd_wrapper = AnomalyNCDWrapper(config_path)
        
        # 运行 NCD 流程
        # 使用关键字参数调用，提高可读性
        result = self.ncd_wrapper.run(
            dataset_path=dataset_path, 
            anomaly_map_path=anomaly_map_path,
            base_data_path=base_path,
            output_dir=output_root # 将根目录传入，内部会创建子目录
        )
        
        return result
