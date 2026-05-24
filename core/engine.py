# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：核心执行引擎 (Core Execution Engine)
# 文件名：engine.py
# 功能：
#   1. 作为系统的总控中枢，协调各子模块（MuSc, DataBridge, AnomalyNCD）的无缝协同。
#   2. 串联并执行全新的结构化批量处理流水线 (BatchPipeline)，彻底告别平铺数据流。
#   3. 实现动态路径路由与跨模块间的资源动态解耦。
# =================================================================================================

import os
import sys

# -------------------------------------------------------------------------------------------------
# 全局路径配置 (Global Path Configuration)
# -------------------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIBS_PATH = os.path.join(PROJECT_ROOT, 'libs')        
MUSC_PATH = os.path.join(LIBS_PATH, 'MuSc')           
NCD_PATH = os.path.join(LIBS_PATH, 'AnomalyNCD')      

sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

class BatchPipeline:
    """
    类：批量处理流水线 (Batch Processing Pipeline)
    
    功能：
        管理并执行从“MuSc结构化特征图生成 -> DataBridge语义分流与格式转换 -> AnomalyNCD高阶聚类发现”的完整流水线。
    """
    
    def __init__(self):
        """
        构造函数：初始化算法包装器状态（采用延迟加载策略，避免初始化时过度占用显存）。
        """
        self.musc_wrapper = None # MuSc 算法封装器实例
        self.ncd_wrapper = None  # AnomalyNCD 算法封装器实例
        
        # 设置默认的结果输出根目录
        self.output_base = os.path.join(PROJECT_ROOT, 'data_store', 'results')
        os.makedirs(self.output_base, exist_ok=True)
        
    def run(self, input_dir, output_dir=None, category_name="custom_dataset"):
        """
        方法：执行完整流水线 (Run Pipeline)
        
        参数:
            input_dir (str): 前端解压后的结构化原图目录（包含 known_normal, known_crack... ）。
            output_dir (str, optional): 结果保存路径。如果不指定，则自动生成带时间戳的目录。
            category_name (str): 动态数据集类别名称，将透传给下游算法。
            
        返回:
            bool: 流水线执行成功返回 True，失败返回 False。
        """
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            output_dir = os.path.join(self.output_base, f"run_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BatchPipeline] Pipeline started with input: {input_dir}")
        print(f"[BatchPipeline] Active Category Name: {category_name}")
        print(f"[BatchPipeline] Target Results Directory: {output_dir}")
        
        # =========================================================================
        # 阶段 1: 运行 MuSc 递归生成结构化异常热力图
        # =========================================================================
        print("\n--- Step 1: Running MuSc (Structural Anomaly Map Generation) ---")
        try:
            musc_maps_tmp = os.path.join(output_dir, 'musc_maps')
            maps_dir = self.run_musc(input_dir, musc_maps_tmp)
            
            if not maps_dir:
                print("[BatchPipeline] Critical Error: MuSc pipeline failed or produced no output.")
                return False
                
            # ==========================================
            # 🚀 核心修复点：强制释放 MuSc 占用的巨量显存
            # ==========================================
            print("[BatchPipeline] Unloading MuSc model and clearing GPU cache...")
            if self.musc_wrapper is not None:
                del self.musc_wrapper       # 删掉 Python 对象引用
                self.musc_wrapper = None    # 重置指针
                
                import torch
                import gc
                gc.collect()                # 强制 Python 垃圾回收
                torch.cuda.empty_cache()    # 强制清空 PyTorch 在 GPU 上的显存缓存
                print("[BatchPipeline] GPU memory successfully released for AnomalyNCD.")
                
        except Exception as e:
            print(f"[BatchPipeline] Exception caught in MuSc step: {e}")
            import traceback
            traceback.print_exc()
            return False

        # =========================================================================
        # 阶段 2: 运行 DataBridge 语义路由分流（核心中枢转换）
        # =========================================================================
        print("\n--- Step 2: Running DataBridge (Semantic Routing & Format Conversion) ---")
        try:
            # 调用重构后的桥接器，动态完成已知/未知异常与正常参考样本的分流
            bridge_result = self.run_bridge(input_dir, maps_dir, output_dir, category_name)
            
            if not bridge_result:
                print("[BatchPipeline] Critical Error: DataBridge semantic routing failed.")
                return False
                
            # 动态提取分流搭建好的 MTD 数据集标准路径
            base_data_path    = bridge_result['base_data_path']     # 提取出的正常参考样本
            dataset_images_path = bridge_result['images_path']        # 待分析原图树
            dataset_maps_path   = bridge_result['anomaly_maps_path']   # 转换后的 PNG 热力图树
            active_category     = bridge_result['category_name']       # 激活的类别名
                
        except Exception as e:
            print(f"[BatchPipeline] Exception caught in DataBridge step: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # =========================================================================
        # 阶段 3: 运行 AnomalyNCD 新类发现（精准投喂）
        # =========================================================================
        print("\n--- Step 3: Running AnomalyNCD (Novel Class Discovery Clustering) ---")
        try: 
            # 彻底废除旧版的硬编码路径，精准投喂从桥接器解耦出来的三个核心路径和动态类别名
            success = self.run_ncd(
                dataset_path=dataset_images_path, 
                anomaly_map_path=dataset_maps_path, 
                base_path=base_data_path, 
                output_root=output_dir,
                category_name=active_category
            )
            
            if not success:
                print("[BatchPipeline] Critical Error: AnomalyNCD clustering step failed.")
                return False
                
        except Exception as e:
            print(f"[BatchPipeline] Exception caught in AnomalyNCD step: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 流程圆满结束后，可以考虑选择性清理临时的 musc_maps 目录以节省服务器空间
        try:
            if os.path.exists(musc_maps_tmp):
                import shutil
                shutil.rmtree(musc_maps_tmp)
                print("[BatchPipeline] Cleaned up temporary MuSc .npy directory.")
        except Exception:
            pass

        print(f"\n[BatchPipeline] Pipeline finished successfully! All aligned results are stored under: {output_dir}")
        return True

    def run_musc(self, input_dir, output_root):
        """
        方法：调用重构后的 MuSc 包装器
        """
        from core.musc_wrapper import MuScWrapper  
        
        config_path = os.path.join(MUSC_PATH, 'configs', 'musc.yaml')
        
        if self.musc_wrapper is None:
            if not os.path.exists(config_path):
                print(f"[Engine] Warning: MuSc Config {config_path} not found.")
            self.musc_wrapper = MuScWrapper(config_path)
            
        saved_paths = self.musc_wrapper.generate_anomaly_maps(input_dir, output_root)
        
        if saved_paths and len(saved_paths) > 0:
            return output_root
        return None

    def run_bridge(self, input_images, input_maps, output_dir, category_name):
        """
        方法：调用重构后的 DataBridge 语义路由算子
        """
        from core.data_bridge import DataBridge 
        
        bridge = DataBridge() 
        print(f"[Engine] Routing and reshaping dataset from {input_images} into standard MTD layout.")
        
        result = bridge.prepare_ncd_dataset(
            raw_images_dir=input_images, 
            maps_dir=input_maps, 
            output_base_dir=output_dir,
            category_name=category_name
        )
        return result  

    def run_ncd(self, dataset_path, anomaly_map_path, base_path, output_root, category_name):
        """
        方法：调用解绑硬编码后的 AnomalyNCD 包装器
        """
        from core.anomalyncd_wrapper import AnomalyNCDWrapper 
        
        config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
        
        if self.ncd_wrapper is None:
             if not os.path.exists(config_path):
                print(f"[Engine] Warning: AnomalyNCD Config {config_path} not found.")
             self.ncd_wrapper = AnomalyNCDWrapper(config_path)
        
        # 将解耦后的参数流、动态类别名完整送入底层训练引擎
        result = self.ncd_wrapper.run(
            dataset_path=dataset_path, 
            anomaly_map_path=anomaly_map_path,
            base_data_path=base_path,
            output_dir=output_root,
            category_name=category_name
        )
        
        return result