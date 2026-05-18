# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：主应用程序入口 (Main Application Entry)
# 文件名：main_app.py
# 功能：
#   1. 构建基于 Streamlit 的 Web 用户界面，作为系统的交互前端。
#   2. 管理系统的不同运行模式（训练、单流推理、批量分析）。
#   3. 处理用户输入（文件上传、参数设置）并调用底层核心引擎 (Core Engine)。
#   4. 实时显示处理进度和结果反馈。
# =================================================================================================

import streamlit as st  # 导入 Streamlit 库，用于构建 Web 界面 (Syntax: import 库名 as 别名)
import sys              # 导入 sys 模块，用于处理 Python 运行时环境
import os               # 导入 os 模块，用于处理文件路径和操作系统交互

# --- 路径配置 (System Path Configuration) ---
# 将项目根目录添加到 Python 的模块搜索路径 (sys.path) 中。
# 作用：确保代码可以从项目根目录导入 'core', 'libs' 等自定义模块。
# 语法详解：
#   __file__: 当前脚本文件的路径变量。
#   os.path.dirname(): 获取路径中的目录部分。
#   os.path.join(..., '..'): 拼接路径，'..' 表示上一级目录。
#   os.path.abspath(): 将相对路径转换为绝对路径。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 页面基础配置 (Page Configuration) ---
# 设置 Streamlit 页面的标题、图标、布局方式等元数据。
# 必须是 Streamlit 命令中的第一个调用。
st.set_page_config(
    page_title="Anomaly Detection System",  # 浏览器标签页标题
    page_icon="🔍",                         # 浏览器标签页图标
    layout="wide",                          # 页面布局模式：'centered' (居中) 或 'wide' (宽屏)
    initial_sidebar_state="expanded",       # 侧边栏初始状态：'expanded' (展开)
)

# --- 界面标题 (UI Header) ---
st.title("Industrial Anomaly Detection System")  # 显示主标题 (H1)
st.markdown("---")                               # 显示水平分割线 (Markdown 语法)

# --- 侧边栏模式选择 (Sidebar Mode Selection) ---
st.sidebar.title("System Mode")  # 侧边栏标题

# 创建单选按钮组，让用户选择系统运行模式。
# st.sidebar.radio 返回用户选中的选项字符串。
mode = st.sidebar.radio(
    "Select Operation Mode:",
    ("Mode 1: Model Training & Registration", 
     "Mode 2: Single Stream Inference", 
     "Mode 3: Batch Analysis (Focus)")
)

# ==============================================================================
# 模式 3: 批量分析 (Batch Analysis) - 核心功能区
# ==============================================================================
if mode == "Mode 3: Batch Analysis (Focus)":
    st.header("Mode 3: Batch Analysis")  # 显示二级标题 (H2)
    
    # 显示信息提示框，说明该模式的功能
    st.info("Upload a folder of images or Select a local directory to process a batch of images for Novel Class Discovery.")
    
    st.write("Provide an input directory OR upload image files directly:") # 显示普通文本
    
    # --- 输入源 1: 本地目录路径 ---
    # 获取默认的输入目录路径：项目根目录/data_store/raw_inputs
    default_input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store", "raw_inputs")
    
    # 创建文本输入框，允许用户手动修改本地目录路径
    input_dir = st.text_input("Local Directory Path:", value=default_input_path)
    
    # --- 输入源 2: 文件上传 ---
    # 创建文件上传组件，允许上传多个文件，限制类型为常见图片格式。
    uploaded_files = st.file_uploader("Or Upload Image Files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    
    # --- 执行按钮 (Action Button) ---
    # st.button 创建一个按钮，当用户点击时返回 True。
    if st.button("Start Batch Pipeline"):
        # 1. 处理上传的文件 (如果存在)
        if uploaded_files:
            # 定义上传文件的临时保存目录
            upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_store", "raw_inputs_uploaded")
            os.makedirs(upload_dir, exist_ok=True) # 创建目录，如果存在则忽略 (exist_ok=True)
            
            # 清空该目录下的旧文件，防止干扰
            for f in os.listdir(upload_dir):
                os.remove(os.path.join(upload_dir, f))
                
            # 将上传的内存文件写入硬盘
            for uploaded_file in uploaded_files:
                # uploaded_file 是一个类似文件的对象
                # uploaded_file.name 获取文件名
                # uploaded_file.getbuffer() 获取文件内容的字节流
                with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # 如果使用了上传功能，将输入目录指向上传目录
            input_dir = upload_dir 
            
        st.write(f"🚀 Starting analysis on: {input_dir}")
        
        # 2. 调用核心处理逻辑
        try:
            # 动态导入核心引擎模块，避免在脚本启动时就加载重型依赖
            from core.engine import BatchPipeline
            
            # 实例化处理流水线对象
            pipeline = BatchPipeline()
            
            # 使用 st.spinner 显示加载动画，直到代码块执行完毕
            with st.spinner('Running Batch Analysis Pipeline... (This may take several minutes)'):
                # 运行流水线，传入输入目录
                success = pipeline.run(input_dir)
                
            # 根据返回结果显示成功或失败信息
            if success:
                st.success("✅ Analysis Complete! Results saved to data_store/results.")
            else:
                st.error("❌ Pipeline Failed. Please check the terminal for error details.")
                
        except ImportError as e:
            # 捕获导入错误 (通常是因为路径问题或依赖未安装)
            st.error(f"Import Error: {e}")
            st.error("Ensure 'core' is in python path and dependencies are installed.")
        except Exception as e:
            # 捕获所有其他运行时异常，并在界面上显示错误信息
            st.error(f"An error occurred: {e}")

# ==============================================================================
# 模式 1: 模型训练 (占位符)
# ==============================================================================
elif mode == "Mode 1: Model Training & Registration":
    st.header("Mode 1: Training & Setup")
    st.warning("This mode is for registering new parts and training the classifier.")

# ==============================================================================
# 模式 2: 实时推理 (占位符)
# ==============================================================================
elif mode == "Mode 2: Single Stream Inference":
    st.header("Mode 2: Live Inference")
    st.warning("This mode is for real-time single image checking.")
