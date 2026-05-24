# -*- coding: utf-8 -*-
# =================================================================================================
# 模块：主应用程序入口 (Main Application Entry)
# 文件名：main_app.py
# 功能：
#   1. 构建基于 Streamlit 的 Web 用户界面。
#   2. 合并训练与批量分析模式（数据上传、后台异步训练、日志可视化）。
#   3. 支持训练后模型的重命名与归档。
#   4. 支持分类结果数据的可视化、在线编辑（修改分类名）与结果报表下载。
# =================================================================================================

import streamlit as st
import sys
import os
import zipfile
import subprocess
import signal
import time
import pandas as pd  # 新增：用于处理结果表单
import shutil

# --- 路径配置 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 状态记录与数据目录
PID_FILE = os.path.join(PROJECT_ROOT, "training_pid.txt")
LOG_FILE = os.path.join(PROJECT_ROOT, "training_log.txt")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_store", "checkpoint")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data_store", "results")

# 确保必要的目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 进程管理辅助函数 ---
def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def get_current_pid():
    if os.path.exists(PID_FILE):
        with open(PID_FILE, "r") as f:
            pid_str = f.read().strip()
            if pid_str.isdigit():
                pid = int(pid_str)
                if is_running(pid):
                    return pid
    return None

# --- 页面基础配置 ---
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Industrial Anomaly Detection System")
st.markdown("---")

# --- 侧边栏模式选择 ---
st.sidebar.title("System Mode")
mode = st.sidebar.radio(
    "Select Operation Mode:",
    (
        "Mode 1: Model Training & Batch Analysis", # 合并后的模式
        "Mode 2: Single Stream Inference"
    )
)

# ==============================================================================
# 模式 1: 训练与批量分析合并模式
# ==============================================================================
if mode == "Mode 1: Model Training & Batch Analysis":
    
    # 建立三个平行的 Tab 标签页，使界面更清晰
    tab_train, tab_model, tab_results = st.tabs([
        "🚀 1. 数据上传与模型训练", 
        "💾 2. 模型保存与管理", 
        "📊 3. 分类结果与评估"
    ])

    # ---------------------------------------------------------
    # Tab 1: 数据上传与控制台
    # ---------------------------------------------------------
    with tab_train:
        st.header("Step 1: Data Preparation & Execution")
        
        # --- 数据输入区 ---
        # 砍掉本地路径输入和多图片零散上传，强制要求结构化的 ZIP 压缩包
        uploaded_file = st.file_uploader(
            "上传包含结构化数据集的 ZIP 压缩包 (如 known_normal, known_crack, unknown_new...)", 
            accept_multiple_files=False, 
            type=['zip']
        )
        
        # --- 进程控制区 ---
        current_pid = get_current_pid()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ 准备数据并开始训练 (Start Pipeline)", use_container_width=True):
                if uploaded_file:
                    upload_dir = os.path.join(PROJECT_ROOT, "data_store", "raw_inputs_uploaded")
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # 彻底清理旧文件和旧文件夹
                    for f in os.listdir(upload_dir):
                        file_path = os.path.join(upload_dir, f)
                        if os.path.isfile(file_path): 
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        
                    with st.spinner('正在处理上传的 ZIP 包并严格规范化目录树...'):
                        # 1. 直接解压，保留原始目录树
                        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                            zip_ref.extractall(upload_dir)
                            
                        # 2. 清理冗余的系统级隐藏文件夹 (如 Mac 压缩包自带的)
                        for root, dirs, files in os.walk(upload_dir):
                            if "__MACOSX" in dirs:
                                shutil.rmtree(os.path.join(root, "__MACOSX"))
                                dirs.remove("__MACOSX")

                        # 3. 严格规范化：自底向上遍历，强制将所有目录和文件名转换为小写
                        # 废弃原有的平铺逻辑，保证 MTD 数据集格式的语义前缀不丢失
                        for root, dirs, files in os.walk(upload_dir, topdown=False):
                            for name in files:
                                lower_name = name.lower()
                                if name != lower_name:
                                    os.rename(os.path.join(root, name), os.path.join(root, lower_name))
                                    
                            for name in dirs:
                                lower_name = name.lower()
                                if name != lower_name:
                                    os.rename(os.path.join(root, name), os.path.join(root, lower_name))

                    input_dir = upload_dir 
                
                    # 4. 清理旧进程并启动新进程
                    if current_pid:
                        try:
                            os.kill(current_pid, signal.SIGTERM)
                            time.sleep(1)
                        except Exception: pass
                    
                    with open(LOG_FILE, "w") as f:
                        f.write(f"Initialize training pipeline on directory: {input_dir}\n")
                        
                    training_command = ["python", "-u", "core/run_pipeline.py", "--input_dir", input_dir]  

                    with open(LOG_FILE, "a") as log_f:
                        process = subprocess.Popen(
                            training_command, 
                            stdout=log_f, 
                            stderr=subprocess.STDOUT,
                            cwd=PROJECT_ROOT
                        )
                    with open(PID_FILE, "w") as f:
                        f.write(str(process.pid))
                        
                    st.success("✅ 结构化数据已就绪，训练流水线已在后台启动！")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠️ 请先上传规范的 ZIP 压缩包！")

        with col2:
            # 中断按钮逻辑保持不变
            if st.button("⏹️ 中断当前训练 (Stop)", type="primary", use_container_width=True):
                if current_pid:
                    try:
                        os.kill(current_pid, signal.SIGTERM)
                        st.warning("⚠️ 训练已强制中断。")
                    except Exception as e:
                        st.error(f"中断失败: {e}")
                    finally:
                        if os.path.exists(PID_FILE): os.remove(PID_FILE)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.info("当前没有正在运行的训练任务。")


        # --- 日志可视化区 ---
        st.subheader("🖥️ 实时训练状态 (Terminal Output)")
        if current_pid:
            st.info(f"🔄 后台进程正在运行中 (PID: {current_pid})...")
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f:
                    logs = f.readlines()
                st.code("".join(logs[-40:]), language="text") # 展示最后40行
                if st.button("刷新日志 (Refresh Logs)"):
                    st.rerun()
        else:
            st.write("💤 任务未运行或已结束。")
            if os.path.exists(LOG_FILE):
                with st.expander("查看最后一次运行的完整日志"):
                    with open(LOG_FILE, "r") as f:
                        st.code(f.read(), language="text")

    # ---------------------------------------------------------
    # Tab 2: 模型管理与重命名
    # ---------------------------------------------------------
    with tab_model:
        st.header("Step 2: Model Management")
        st.write(f"当前模型存储路径: `{MODEL_DIR}`")
        
        # 获取 checkpoint 目录下所有文件
        model_files = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]
        
        if model_files:
            selected_model = st.selectbox("选择要管理/重命名的模型文件:", model_files)
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                new_model_name = st.text_input("输入新的模型名称 (包含后缀，如 best_model.pt):", value=selected_model)
            
            with col_m2:
                st.write("") # 占位对齐
                st.write("") # 占位对齐
                if st.button("💾 重命名并保存", type="primary"):
                    if new_model_name and new_model_name != selected_model:
                        old_path = os.path.join(MODEL_DIR, selected_model)
                        new_path = os.path.join(MODEL_DIR, new_model_name)
                        try:
                            os.rename(old_path, new_path)
                            st.success(f"✅ 成功将 `{selected_model}` 重命名为 `{new_model_name}`")
                            time.sleep(1)
                            st.rerun() # 刷新下拉列表
                        except Exception as e:
                            st.error(f"重命名失败: {e}")
                    else:
                        st.warning("名称未改变或为空。")
        else:
            st.info("暂未在 checkpoint 目录下检测到模型文件。请先完成训练。")

    # ---------------------------------------------------------
    # Tab 3: 分类结果编辑与下载
    # ---------------------------------------------------------
    with tab_results:
        st.header("Step 3: Classification Results Review")
        
        # 假设你的后端代码会将最终结果保存为一个 CSV 文件，这里我们需要定位它
        # 实际情况中你需要确保你的后端能生成类似于 data_store/results/predictions.csv 的文件
        csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
        
        if csv_files:
            selected_csv = st.selectbox("选择结果报表:", csv_files)
            csv_path = os.path.join(RESULTS_DIR, selected_csv)
            
            try:
                # 读取 CSV 数据
                df = pd.read_csv(csv_path)
                st.write("👉 **提示:** 双击表格中的单元格即可直接修改分类名称或数据。")
                
                # 使用 st.data_editor 支持前端在线编辑数据表
                edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
                
                # 将编辑后的数据转换回 CSV 格式供下载
                csv_buffer = edited_df.to_csv(index=False).encode('utf-8-sig') # utf-8-sig 兼容 Excel 中文
                
                col_d1, col_d2 = st.columns([1, 3])
                with col_d1:
                    st.download_button(
                        label="📥 下载更新后的报表",
                        data=csv_buffer,
                        file_name=f"updated_{selected_csv}",
                        mime="text/csv"
                    )
                with col_d2:
                    if st.button("💾 覆盖保存到服务器"):
                        edited_df.to_csv(csv_path, index=False)
                        st.success("✅ 服务器端文件已更新！")
                        
            except Exception as e:
                st.error(f"读取或解析 CSV 文件失败: {e}")
        else:
            st.info("暂未在 results 目录下检测到结果表单 (CSV文件)。请等待分析流程完成，或确保后端代码正确生成了该文件。")


# ==============================================================================
# 模式 2: 实时推理
# ==============================================================================
elif mode == "Mode 2: Single Stream Inference":
    st.header("Mode 2: Live Inference")
    st.warning("This mode is for real-time single image checking.")