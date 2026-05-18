# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：核心执行引�? (Core Execution Engine)
# 文件名：engine.py
# 功能�?
#   1. 作为系统的中枢控制器，协调各�?子模块（MuSc, DataBridge, AnomalyNCD）的协同工作�?
#   2. 定义并执行批量�?�理流水�? (BatchPipeline)，�?�理数据流的传递�?
#   3. 处理模块间的�?径映射和资源初�?�化�?
# =================================================================================================

import os
import sys
import shutil

# -------------------------------------------------------------------------------------------------
# 全局�?径配�? (Global Path Configuration)
# -------------------------------------------------------------------------------------------------
# 动态获取项�?根目录，�?保代码在不同�?境下都能正确找到资源�?
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 获取当前文件的上上级�?�?

# 定义关键子目录路�?
LIBS_PATH = os.path.join(PROJECT_ROOT, 'libs')        # �?三方�?/算法库目�?
MUSC_PATH = os.path.join(LIBS_PATH, 'MuSc')           # MuSc 算法�?�?
NCD_PATH = os.path.join(LIBS_PATH, 'AnomalyNCD')      # AnomalyNCD 算法�?�?

# �? core �?录加入系统路径，使其他模块可以�?�入 core 包下的内�?
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

class BatchPipeline:
    """
    类：批量处理流水�? (Batch Processing Pipeline)
    
    功能�?
        封�?�了完整的异常�?�测和新类发现流程�?
        实现�? "MuSc生成�?力图 -> DataBridge格式�?�? -> AnomalyNCD聚类分析" 的逻辑串联�?
    """
    
    def __init__(self):
        """
        构造函数：初�?�化流水线状态�?
        注意：Wrapper 类采用延迟初始化 (Lazy Loading) 策略，初始�?�为 None�?
        好�?�是�?动引擎时不需要立即加载显存占用大的模型，�?有在真�?�运行时才加载�?
        """
        self.musc_wrapper = None # MuSc 算法封�?�器实例
        self.ncd_wrapper = None  # AnomalyNCD 算法封�?�器实例
        
        #设置默�?�的结果输出根目录：data_store/results
        self.output_base = os.path.join(PROJECT_ROOT, 'data_store', 'results')
        # �?保输出目录存�?，�?�果不存在则创建
        os.makedirs(self.output_base, exist_ok=True)
        
    def run(self, input_dir, output_dir=None):
        """
        方法：执行完整流水线 (Run Pipeline)
        
        参数:
            input_dir (str): 输入�?录，包含待�?�测的原�?�图像�?
            output_dir (str, optional): 结果保存�?径。�?�果不指定，则自动生成带时间戳的�?录�?
            
        返回:
            bool: 流水线执行成功返�? True，失败返�? False�?
        """
        import time
        # 生成格式化的时间戳字符串 (例�??: 20231027_103000)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 如果�?指定输出�?录，则在 base �?录下创建一�?新目�?
        if output_dir is None:
            output_dir = os.path.join(self.output_base, f"run_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BatchPipeline] Pipeline started on {input_dir}")
        print(f"[BatchPipeline] Results will be saved to {output_dir}")
        
        # =========================================================================
        # 阶�?? 1: 运�?? MuSc 生成异常�?力图 (Anomaly Map Generation)
        # =========================================================================
        print("\n--- Step 1: Running MuSc (Anomaly Map Generation) ---")
        try:
            # 调用内部方法 run_musc
            # 输入：原始图片目�?
            # 输出：生成的 .npy �?力图所在的�?录路�?
            maps_dir = self.run_musc(input_dir, output_dir)
            
            # 校验结果：�?�果�?径为空，说明 MuSc 运�?�失败或�?生成文件
            if not maps_dir:
                print("[BatchPipeline] MuSc failed or produced no output.")
                return False
        except Exception as e:
            # 异常处理：打印错�?并打印堆栈跟�?，方便调�?
            print(f"[BatchPipeline] Error in MuSc step: {e}")
            import traceback
            traceback.print_exc()
            return False

        # =========================================================================
        # 阶�?? 2: 运�?? DataBridge 数据�?�? (Format Conversion)
        # =========================================================================
        print("\n--- Step 1.5: Running DataBridge (Format Conversion) ---")
        try:
            # 定义�?间数�?的存储目�?
            interim_dir = os.path.join(output_dir, 'interim_ncd_data')
            
            # 调用 run_bridge 进�?�转�?
            # 作用：将 MuSc 的输�? (.npy) �?�?�? AnomalyNCD 需要的�?录结构和格式 (.png)
            # 返回值：一�?字典，包�?�?换后的图像路径和�?力图�?�?
            bridge_result = self.run_bridge(input_dir, maps_dir, interim_dir)
            
            if not bridge_result:
                print("[BatchPipeline] DataBridge failed.")
                return False
                
            # 提取�?换后的路径，准�?�传给下一�?
            dataset_images_path = bridge_result['images_path']
            dataset_maps_path   = bridge_result['anomaly_maps_path']
                
        except Exception as e:
            print(f"[BatchPipeline] Error in DataBridge step: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # =========================================================================
        # 阶�?? 3: 运�?? AnomalyNCD 新类发现 (Novel Class Discovery)
        # =========================================================================
        print("\n--- Step 2: Running AnomalyNCD (Novel Class Discovery) ---")
        try: 
            # 指定正常样本的参考数�?集路�? (Normal Reference Data)
            # AnomalyNCD 算法通常需要�?�比正常样本来区分异常类型�?
            base_data_path = os.path.join(PROJECT_ROOT, 'data_store', 'raw_inputs', 'normal_ref') 
            
            # 防御性编程：如果参考目录不存在，创建一�?空目录防止程序崩�?
            # 注意：在实际生产�?境中，这里应该报错或提示用户提供正常样本
            if not os.path.exists(base_data_path):
                 print(f"[BatchPipeline] Warning: Base data path {base_data_path} not found. Creating empty.")
                 os.makedirs(base_data_path, exist_ok=True)
            
            # 调用 run_ncd 执�?�核心聚类逻辑
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
        方法：封�? MuSc 算法的调用逻辑
        
        返回:
            str: 生成的异常图 (.npy 文件) 所在的具体�?录路径，失败返回 None�?
        """
        from core.musc_wrapper import MuScWrapper  # 局部�?�入，避免循�?依赖
        
        # 配置文件�?�?
        config_path = os.path.join(MUSC_PATH, 'configs', 'musc.yaml')
        # 输出子目�?
        maps_output_dir = os.path.join(output_root, 'anomaly_maps')
        
        # 懒加载：如果�?�?一次调�?，则初�?�化 Wrapper 实例
        if self.musc_wrapper is None:
            if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
            # 初�?�化 MuSc 模型 (加载权重�? GPU/CPU)
            self.musc_wrapper = MuScWrapper(config_path)
            
        # 执�?�批量生�?
        saved_paths = self.musc_wrapper.generate_anomaly_maps(input_dir, maps_output_dir)
        
        # 检查是否生成了文件
        if saved_paths and len(saved_paths) > 0:
            return maps_output_dir
        return None

    def run_bridge(self, input_images, input_maps, output_dir):
        """
        方法：封�? DataBridge 的调用逻辑
        
        参数:
            input_images: 原�?�图片目�?
            input_maps: MuSc 生成�? .npy �?力图�?�?
            output_dir: �?换后数据的存放目�?
        """
        from core.data_bridge import DataBridge # 局部�?�入
        
        bridge = DataBridge() 
        
        print(f"[Engine] Bridging data from {input_images} to {output_dir}")
        # 执�?�转�?
        result = bridge.prepare_ncd_dataset(input_images, input_maps, output_dir)
        return result  

    def run_ncd(self, dataset_path, anomaly_map_path, base_path, output_root):
        """
        方法：封�? AnomalyNCD 算法的调用逻辑
        
        参数:
            dataset_path: 整理好的图片�?�? (DataBridge 输出)
            anomaly_map_path: 整理好的�?力图�?�? (DataBridge 输出)
            base_path: 正常样本�?�?
            output_root: 结果根目�?
        """
        from core.anomalyncd_wrapper import AnomalyNCDWrapper # 局部�?�入
        
        config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
        ncd_output_dir = os.path.join(output_root, 'ncd_results')
        
        # 懒加载初始化
        if self.ncd_wrapper is None:
             if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
             self.ncd_wrapper = AnomalyNCDWrapper(config_path)
        
        # 运�?? NCD 流程
        # 使用关键字参数调�?，提高可读�?
        result = self.ncd_wrapper.run(
            dataset_path=dataset_path, 
            anomaly_map_path=anomaly_map_path,
            base_data_path=base_path,
            output_dir=output_root # 将根�?录传入，内部会创建子�?�?
        )
        
        return result
