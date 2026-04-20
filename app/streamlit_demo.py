# -*- coding: utf-8 -*-
"""Streamlit 前端演示页。

功能：
1. 提供三种模式的入口（当前重点展示 Mode 3 批处理）。
2. 负责文件上传（ZIP 或多图）并准备输入目录。
3. 调用 `core.engine.BatchPipeline` 执行后端流水线。
"""

import os
import shutil
import sys
import zipfile

import streamlit as st

# 加入项目根目录，确保可导入 `core` 包。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Industrial Anomaly Detection System")
st.markdown("---")

st.sidebar.title("System Mode")
mode = st.sidebar.radio(
    "Select Operation Mode:",
    (
        "Mode 1: Model Training & Registration",
        "Mode 2: Single Stream Inference",
        "Mode 3: Batch Analysis (Focus)",
    ),
)


if mode == "Mode 3: Batch Analysis (Focus)":
    st.header("Mode 3: Batch Analysis")
    st.info("上传包含数据集的 ZIP（推荐）或直接上传多张图像，系统将执行批次异常识别。")

    uploaded_files = st.file_uploader(
        "Upload Image Dataset (ZIP) or Images",
        accept_multiple_files=True,
        type=["zip", "png", "jpg", "jpeg", "bmp", "tiff"],
    )

    if st.button("Start Batch Pipeline"):
        if not uploaded_files:
            st.error("请先上传 ZIP 文件或图像文件。")
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            upload_root = os.path.join(project_root, "data_store", "raw_inputs_uploaded")

            # 每次执行前清理旧输入，避免残留数据影响结果。
            if os.path.exists(upload_root):
                shutil.rmtree(upload_root)
            os.makedirs(upload_root, exist_ok=True)

            input_dir = upload_root
            zip_files = [file_item for file_item in uploaded_files if file_item.name.lower().endswith(".zip")]

            if zip_files:
                # 仅取第一个 ZIP 作为本次批处理输入。
                zip_file = zip_files[0]
                zip_path = os.path.join(upload_root, zip_file.name)

                with open(zip_path, "wb") as save_file:
                    save_file.write(zip_file.getbuffer())

                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(upload_root)

                    st.success(f"已解压：{zip_file.name}")

                    extracted_dirs = [
                        item
                        for item in os.listdir(upload_root)
                        if os.path.isdir(os.path.join(upload_root, item))
                    ]
                    root_has_images = any(
                        item.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                        for item in os.listdir(upload_root)
                    )
                    if len(extracted_dirs) == 1 and not root_has_images:
                        input_dir = os.path.join(upload_root, extracted_dirs[0])
                        st.info(f"自动识别数据根目录：{input_dir}")

                except zipfile.BadZipFile:
                    st.error("上传文件不是有效的 ZIP。")
                    st.stop()

                os.remove(zip_path)

            else:
                # 如果不是 ZIP，则按“散图批处理”保存到同一目录。
                for uploaded_file in uploaded_files:
                    target_path = os.path.join(upload_root, uploaded_file.name)
                    with open(target_path, "wb") as image_file:
                        image_file.write(uploaded_file.getbuffer())
                st.success(f"已保存 {len(uploaded_files)} 张图像。")

            st.write("开始执行批处理流水线...")

            try:
                from core.engine import BatchPipeline

                pipeline = BatchPipeline()
                with st.spinner("Running Batch Analysis Pipeline... (This may take several minutes)"):
                    success = pipeline.run(input_dir)

                if success:
                    st.success("分析完成，结果已保存到 data_store/results。")
                    st.balloons()
                else:
                    st.error("流水线执行失败，请查看终端日志。")

            except ImportError as import_error:
                st.error(f"Import Error: {import_error}")
                st.error("请确认 `core` 包可导入且依赖已安装。")
            except Exception as run_error:
                st.error(f"发生错误：{run_error}")

elif mode == "Mode 1: Model Training & Registration":
    st.header("Mode 1: Training & Setup")
    st.warning("该模式用于新类别注册与训练流程触发（当前为占位页）。")

elif mode == "Mode 2: Single Stream Inference":
    st.header("Mode 2: Live Inference")
    st.warning("该模式用于单图实时检测（当前为占位页）。")
