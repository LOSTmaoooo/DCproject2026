import streamlit as st
import sys
import os

# =================================================================================================
# 模块：主应用程序 (Streamlit Frontend)
# 功能：提供用户交互界面（Web UI），允许用户选择模式、上传数据并触发核心引擎进行分析。
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 环境配置
# -------------------------------------------------------------------------------------------------
# 将项目根目录添加到 python path，确保可以导入 app 同级或上级的 core 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Streamlit 页面配置
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="?",
    layout="wide", # 使用宽屏布局
    initial_sidebar_state="expanded",
)

# 页面标题
st.title("Industrial Anomaly Detection System")
st.markdown("---")

# -------------------------------------------------------------------------------------------------
# 侧边栏导航 (Sidebar Navigation)
# -------------------------------------------------------------------------------------------------
st.sidebar.title("System Mode")
mode = st.sidebar.radio(
    "Select Operation Mode:",
    ("Mode 1: Model Training & Registration", 
     "Mode 2: Single Stream Inference", 
     "Mode 3: Batch Analysis (Focus)")
)

# -------------------------------------------------------------------------------------------------
# 模式 3: 批量分析 (主要功能)
# -------------------------------------------------------------------------------------------------
if mode == "Mode 3: Batch Analysis (Focus)":
    st.header("Mode 3: Batch Analysis")
    st.info("Upload a folder of images or Select a local directory to process a batch of images for Novel Class Discovery.")
    
    # 输入相关 UI
    # 目前使用文本框输入本地路径，未来可扩展为文件夹选择器
    # 默认值是为了方便测试
    input_dir = st.text_input("Input Directory Path:", value="d:/XYH/DCproject/data_store/raw_inputs")
    
    # 触发按钮
    if st.button("Start Batch Pipeline"):
        st.write(f"? Starting analysis on: {input_dir}")
        
        try:
            # 在按钮点击后才导入引擎，减少启动时间
            from core.engine import BatchPipeline
            
            # 实例化流水线对象
            pipeline = BatchPipeline()
            
            # 显示加载动画
            with st.spinner('Running Batch Analysis Pipeline... (This may take several minutes)'):
                # 运行完整流水线
                # output_dir 默认为空，系统会自动生成带时间戳的文件夹
                success = pipeline.run(input_dir)
                
            # 结果反馈
            if success:
                st.success("? Analysis Complete! Results saved to data_store/results.")
            else:
                st.error("? Pipeline Failed. Please check the terminal for error details.")
                
        except ImportError as e:
            st.error(f"Import Error: {e}")
            st.error("Ensure 'core' is in python path and dependencies are installed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# -------------------------------------------------------------------------------------------------
# 模式 1: 与 模式 2 (预留/开发中)
# -------------------------------------------------------------------------------------------------
elif mode == "Mode 1: Model Training & Registration":
    st.header("Mode 1: Training & Setup")
    st.warning("This mode is for registering new parts and training the classifier.")

elif mode == "Mode 2: Single Stream Inference":
    st.header("Mode 2: Live Inference")
    st.warning("This mode is for real-time single image checking.")
