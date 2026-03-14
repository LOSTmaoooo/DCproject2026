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
    st.write("Provide an input directory OR upload image files directly:")
    
    # 选项1: 本地路径输入 (Local Path Input)
    input_dir = st.text_input("Local Directory Path:", value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store", "raw_inputs"))
    
    # 选项2: 多文件上传 (Multiple files upload)
    uploaded_files = st.file_uploader("Or Upload Image Files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    
    # 触发按钮
    if st.button("Start Batch Pipeline"):
        # 如果用户上传了文件，先保存到一个本地临时目录中
        if uploaded_files:
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store", "raw_inputs_uploaded")
            os.makedirs(upload_dir, exist_ok=True)
            # 清空旧数据以防混淆
            for f in os.listdir(upload_dir):
                os.remove(os.path.join(upload_dir, f))
                
            for uploaded_file in uploaded_files:
                with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
            input_dir = upload_dir # 使用我们刚刚保存了上传文件的路径
            
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
