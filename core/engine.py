import os
import sys
import shutil

# =================================================================================================
# 模块：核心执行引擎 (Core Execution Engine)
# 功能：协调系统的各个组件（MuSc, AnomalyNCD 等），串联成完整的处理流水线。
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 路径配置
# -------------------------------------------------------------------------------------------------
# 获取项目根目录，以便动态定位其他资源和库
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS_PATH = os.path.join(PROJECT_ROOT, 'libs')
MUSC_PATH = os.path.join(LIBS_PATH, 'MuSc')
NCD_PATH = os.path.join(LIBS_PATH, 'AnomalyNCD')

# 将 core 目录加入系统路径，确保可以从其他地方(如 Streamlit app) 导入 core 下的模块
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

class BatchPipeline:
    """
    批量处理流水线 (Batch Processing Pipeline)
    
    该类负责执行 'Mode 3: Batch Analysis' 的主要业务逻辑。
    它按顺序调用以下步骤：
    1. 使用 MuSc 生成异常图 (Anomaly Maps)。
    2. 使用 AnomalyNCD 进行新类发现 (Novel Class Discovery)，基于生成的异常图和原始图像进行聚类分析。
    """
    
    def __init__(self):
        """
        初始化流水线。
        Wrapper 类被延迟初始化（设为 None），以节省启动资源，仅在需要时加载模型。
        """
        self.musc_wrapper = None
        self.ncd_wrapper = None
        
        #设置默认的输出目录：data_store/results
        self.output_base = os.path.join(PROJECT_ROOT, 'data_store', 'results')
        os.makedirs(self.output_base, exist_ok=True)
        
    def run(self, input_dir, output_dir=None):
        """
        执行完整流水线的主入口方法。
        
        参数:
            input_dir (str): 包含待分析图像（可能有异常）的输入目录路径。
            output_dir (str, optional): 结果保存路径。如果为 None，则自动生成带时间戳的文件夹。
            
        返回:
            bool: 如果流水线成功完成返回 True，否则返回 False。
        """
        import time
        # 生成时间戳，用于创建唯一的运行输出文件夹
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = os.path.join(self.output_base, f"run_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BatchPipeline] Pipeline started on {input_dir}")
        print(f"[BatchPipeline] Results will be saved to {output_dir}")
        
        # -------------------------------------------------------------------------
        # 步骤 1: 运行 MuSc 生成异常图
        # -------------------------------------------------------------------------
        print("\n--- Step 1: Running MuSc (Anomaly Map Generation) ---")
        try:
            # 调用 run_musc 方法，返回生成的异常图所在目录
            # Output: output_dir/anomaly_maps
            maps_dir = self.run_musc(input_dir, output_dir)
            if not maps_dir:
                print("[BatchPipeline] MuSc failed or produced no output.")
                return False
        except Exception as e:
            # 捕获并打印详细错误堆栈，防止整个程序崩溃
            print(f"[BatchPipeline] Error in MuSc step: {e}")
            import traceback
            traceback.print_exc()
            return False

        # -------------------------------------------------------------------------
        # 步骤 1.5: 运行 DataBridge (数据格式转换与预处理)
        # -------------------------------------------------------------------------
        print("\n--- Step 1.5: Running DataBridge (Format Conversion) ---")
        try:
            # 定义中间数据存储目录
            interim_dir = os.path.join(output_dir, 'interim_ncd_data')
            
            # 使用 DataBridge 进行转换
            # 它负责将 MuSc 的 .npy 热力图转为 NCD 需要的 dataset 结构 (images/masks)
            dataset_ready_path = self.run_bridge(input_dir, maps_dir, interim_dir)
            
            if not dataset_ready_path:
                print("[BatchPipeline] DataBridge failed.")
                return False
                
        except Exception as e:
            print(f"[BatchPipeline] Error in DataBridge step: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # -------------------------------------------------------------------------
        # 步骤 2: 运行 AnomalyNCD 进行新类发现
        # -------------------------------------------------------------------------
        print("\n--- Step 2: Running AnomalyNCD (Novel Class Discovery) ---")
        try: 
            # 定义基础/正常类别的参考数据集路径。
            # AnomalyNCD 通常需要对比正常样本来发现新类别。
            # 这里的路径目前是硬编码的示例，实际使用中可能需要配置或作为参数传入。
            base_data_path = os.path.join(PROJECT_ROOT, 'data_store', 'raw_inputs', 'normal_ref') 
            
            # 如果参考目录不存在，创建一个空目录以防报错（在实际部署中应确保有数据）
            if not os.path.exists(base_data_path):
                 print(f"[BatchPipeline] Warning: Base data path {base_data_path} not found. Creating empty.")
                 os.makedirs(base_data_path, exist_ok=True)
            
            # 调用 run_ncd 方法执行聚类分析
            # 注意：这里的输入 dataset_path 变成了 DataBridge 处理后的路径
            # maps_dir 依然传入，以备 NCD 内部需要原始 heatmaps
            success = self.run_ncd(dataset_ready_path, maps_dir, base_data_path, output_dir)
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
        封装 MuSc 的调用逻辑。
        
        参数:
            input_dir (str): 输入图像目录。
            output_root (str): 本次运行的根输出目录。
            
        返回:
            str: 生成的异常图（.npy 文件）所在的具体目录路径。
        """
        # 延迟导入 MuScWrapper，避免在文件顶部导入时的循环依赖风险
        from core.musc_wrapper import MuScWrapper
        
        # 配置 MuSc 的配置文件路径
        config_path = os.path.join(MUSC_PATH, 'configs', 'musc.yaml')
        # 定义异常图的特定输出子目录
        maps_output_dir = os.path.join(output_root, 'anomaly_maps')
        
        # 懒加载：如果是第一次运行，则初始化 MuScWrapper
        if self.musc_wrapper is None:
            # We need to ensure config exists or handle error
            if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
                
            self.musc_wrapper = MuScWrapper(config_path)
            
        # 执行生成过程
        # generate_anomaly_maps 返回生成的文件列表
        saved_paths = self.musc_wrapper.generate_anomaly_maps(input_dir, maps_output_dir)
        
        # 如果成功生成了文件，返回目录路径，否则返回 None
        if saved_paths and len(saved_paths) > 0:
            return maps_output_dir
        return None

    def run_bridge(self, input_images, input_maps, output_dir):
        """
        封装 DataBridge 的调用逻辑。
        将 MuSc 的输出转换为 AnomalyNCD 的输入格式。
        """
        from core.data_bridge import DataBridge
        
        bridge = DataBridge() # 使用默认阈值，如有需要可传入参数
        
        print(f"[Engine] Bridging data from {input_images} to {output_dir}")
        # data_bridge.prepare_ncd_dataset 应该返回处理后的有效数据目录
        # 假设它返回的是 output_dir 或者 output_dir 下的具体子目录
        # 根据 data_bridge.py 的逻辑，它会创建 output_dir/unknown_batch/images 等
        
        bridge.prepare_ncd_dataset(input_images, input_maps, output_dir)
        
        # 由于 AnomalyNCD 可能需要只要 root 目录，或者具体的子目录
        # 这里我们返回 output_dir，并在 wrapper 中处理具体路径，或者这里返回具体子目录
        # 假设 prepare_ncd_dataset 创建了标准结构，我们返回 output_dir 即可
        return output_dir

    def run_ncd(self, dataset_path, anomaly_map_path, base_path, output_root):
        """
        封装 AnomalyNCD 的调用逻辑。
        
        参数:
            dataset_path (str): 包含待发现类别的图像路径（即本次的输入）。
            anomaly_map_path (str): 上一步 MuSc 生成的异常图路径。
            base_path (str): 正常样本的参考路径。
            output_root (str): 本次运行的根输出目录。
            
        返回:
            bool: 执行成功返回 True。
        """
        from core.anomalyncd_wrapper import AnomalyNCDWrapper
        
        # 配置 AnomalyNCD 的配置文件路径
        config_path = os.path.join(NCD_PATH, 'configs', 'AnomalyNCD.yaml')
        # 定义 NCD 结果的保存路径
        ncd_output_dir = os.path.join(output_root, 'ncd_results')
        
        # 懒加载初始化
        if self.ncd_wrapper is None:
             if not os.path.exists(config_path):
                print(f"[Engine] Warning: Config {config_path} not found.")
             self.ncd_wrapper = AnomalyNCDWrapper(config_path)
        
        # 调用 Wrapper 的 run 方法
        # 将 output_root 传递给 wrapper，以便它能在其下创建结构化的子目录
        result = self.ncd_wrapper.run(
            dataset_path=dataset_path, 
            anomaly_map_path=anomaly_map_path,
            base_data_path=base_path,
            output_dir=output_root # Use root so NCD structures inside it
        )
        
        return result
