# -*- coding: utf-8 -*-
"""Streamlit 前端演示页。"""

import csv
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# 加入项目根目录，确保可导入 `core` 包。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_EXTS = (".pt", ".pth", ".ckpt", ".bin")
UNKNOWN_THRESHOLD = 0.5


def _collect_images_recursive(root_dir: str) -> list[str]:
    image_paths: list[str] = []
    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(IMAGE_EXTS):
                image_paths.append(os.path.join(current_root, file_name))
    image_paths.sort()
    return image_paths


def _collect_models_recursive(models_root: str) -> list[str]:
    model_paths: list[str] = []
    if not os.path.isdir(models_root):
        return model_paths

    for current_root, _, files in os.walk(models_root):
        for file_name in files:
            if file_name.lower().endswith(MODEL_EXTS):
                model_paths.append(os.path.join(current_root, file_name))

    model_paths.sort()
    return model_paths


def _safe_score_from_map(map_path: str) -> float:
    import numpy as np

    if not os.path.exists(map_path):
        return 0.0
    anomaly_map = np.load(map_path)
    if anomaly_map.size == 0:
        return 0.0
    max_value = float(anomaly_map.max())
    if max_value > 1.0:
        max_value = max_value / 255.0
    return float(max(0.0, min(1.0, max_value)))


def _score_to_cluster(score: float) -> int:
    return 1 if score >= UNKNOWN_THRESHOLD else 0


def _write_summary(rows: list[dict], output_dir: str, file_prefix: str = "results_summary") -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
    json_path = os.path.join(output_dir, f"{file_prefix}.json")

    fields = [
        "image",
        "input_path",
        "score",
        "cluster_id",
        "anomaly_type",
        "is_unknown",
        "anomaly_map_path",
        "model_path",
        "latency_ms",
        "status",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, ensure_ascii=False, indent=2)

    return csv_path, json_path


def _build_summary_from_input_and_maps(input_dir: str, maps_dir: str, model_path: str = "") -> list[dict]:
    rows: list[dict] = []
    image_paths = _collect_images_recursive(input_dir)
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_stem = Path(image_name).stem
        map_path = os.path.join(maps_dir, f"{image_stem}_map.npy")
        score = _safe_score_from_map(map_path)
        cluster_id = _score_to_cluster(score)
        rows.append(
            {
                "image": image_name,
                "input_path": image_path,
                "score": round(score, 6),
                "cluster_id": cluster_id,
                "anomaly_type": str(cluster_id),
                "is_unknown": bool(score >= UNKNOWN_THRESHOLD and cluster_id == -1),
                "anomaly_map_path": map_path if os.path.exists(map_path) else "",
                "model_path": model_path,
                "latency_ms": "",
                "status": "ok" if os.path.exists(map_path) else "map_missing",
                "error": "" if os.path.exists(map_path) else "anomaly map not found",
            }
        )
    return rows


def _group_by_anomaly_type(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        anomaly_type = str(row.get("anomaly_type", "unknown"))
        grouped.setdefault(anomaly_type, []).append(row)
    return grouped


def _apply_rename_map(rows: list[dict], rename_map: dict[str, str]) -> list[dict]:
    renamed_rows: list[dict] = []
    for row in rows:
        raw_type = str(row.get("anomaly_type", "unknown"))
        new_row = dict(row)
        new_row["anomaly_type"] = rename_map.get(raw_type, raw_type)
        renamed_rows.append(new_row)
    return renamed_rows


@st.cache_resource(show_spinner=False)
def _load_musc_wrapper(config_path: str):
    from core.musc_wrapper import MuScWrapper

    return MuScWrapper(config_path=config_path)


def _run_single_image_pipeline(single_image_path: str, project_root: str, model_path: str = "") -> dict:
    single_output_dir = os.path.join(
        project_root,
        "data_store",
        "results",
        f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    maps_dir = os.path.join(single_output_dir, "anomaly_maps")
    os.makedirs(maps_dir, exist_ok=True)

    config_path = os.path.join(project_root, "libs", "MuSc", "configs", "musc.yaml")
    wrapper = _load_musc_wrapper(config_path)

    start_time = time.perf_counter()
    wrapper.generate_anomaly_maps(os.path.dirname(single_image_path), maps_dir)
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    image_name = os.path.basename(single_image_path)
    image_stem = Path(image_name).stem
    map_path = os.path.join(maps_dir, f"{image_stem}_map.npy")

    score = _safe_score_from_map(map_path)
    cluster_id = _score_to_cluster(score)
    return {
        "image": image_name,
        "score": round(score, 6),
        "cluster_id": cluster_id,
        "anomaly_type": str(cluster_id),
        "is_unknown": bool(score >= UNKNOWN_THRESHOLD and cluster_id == -1),
        "paths": {
            "input": single_image_path,
            "anomaly_map": map_path if os.path.exists(map_path) else "",
        },
        "meta": {
            "model_path": model_path,
            "threshold": UNKNOWN_THRESHOLD,
            "latency_ms": latency_ms,
            "output_dir": single_output_dir,
        },
    }


def _render_rename_and_browser(rows: list[dict], result_dir: str):
    if not rows:
        st.warning("当前没有可浏览的结果记录。")
        return

    grouped = _group_by_anomaly_type(rows)
    st.subheader("异常分类浏览与重命名")
    st.write(f"当前异常类型数量：{len(grouped)}")

    rename_map: dict[str, str] = st.session_state.get("anomaly_rename_map", {})
    with st.expander("重命名异常类型", expanded=True):
        for old_type in sorted(grouped.keys()):
            default_name = rename_map.get(old_type, old_type)
            rename_map[old_type] = st.text_input(
                f"类型 {old_type} 的新名称",
                value=default_name,
                key=f"rename_{old_type}",
            )

        if st.button("应用重命名并保存清单"):
            st.session_state["anomaly_rename_map"] = rename_map
            renamed_rows = _apply_rename_map(rows, rename_map)
            named_csv, named_json = _write_summary(renamed_rows, result_dir, "results_summary_named")
            st.session_state["latest_rows"] = renamed_rows
            st.success("重命名已应用并保存。")
            st.write(f"CSV: {named_csv}")
            st.write(f"JSON: {named_json}")

    rows_to_show = st.session_state.get("latest_rows", rows)
    grouped_show = _group_by_anomaly_type(rows_to_show)

    with st.expander("按异常类型浏览图片", expanded=True):
        max_preview = st.slider("每类最多展示图片数", min_value=1, max_value=30, value=8, step=1)
        for anomaly_type, type_rows in sorted(grouped_show.items(), key=lambda item: item[0]):
            st.markdown(f"### 类型：{anomaly_type}（{len(type_rows)} 张）")
            preview_rows = type_rows[:max_preview]
            columns = st.columns(4)
            for index, row in enumerate(preview_rows):
                column = columns[index % 4]
                with column:
                    image_path = row.get("input_path", "")
                    if image_path and os.path.exists(image_path):
                        st.image(Image.open(image_path), caption=f"{row['image']} | score={row['score']}")
                    else:
                        st.caption(f"{row['image']}（源图不可用）")


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
                output_dir = os.path.join(
                    project_root,
                    "data_store",
                    "results",
                    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
                with st.spinner("Running Batch Analysis Pipeline... (This may take several minutes)"):
                    success = pipeline.run(input_dir, output_dir=output_dir)

                if success:
                    st.success("分析完成，结果已保存到 data_store/results。")
                    st.balloons()

                    maps_dir = os.path.join(output_dir, "anomaly_maps")
                    summary_rows = _build_summary_from_input_and_maps(input_dir, maps_dir)
                    summary_csv, summary_json = _write_summary(summary_rows, output_dir, "results_summary")

                    st.session_state["latest_rows"] = summary_rows
                    st.session_state["latest_result_dir"] = output_dir
                    st.write(f"结果目录：{output_dir}")
                    st.write(f"CSV 清单：{summary_csv}")
                    st.write(f"JSON 清单：{summary_json}")
                else:
                    st.error("流水线执行失败，请查看终端日志。")

            except ImportError as import_error:
                st.error(f"Import Error: {import_error}")
                st.error("请确认 `core` 包可导入且依赖已安装。")
            except Exception as run_error:
                st.error(f"发生错误：{run_error}")

    latest_rows = st.session_state.get("latest_rows", [])
    latest_result_dir = st.session_state.get("latest_result_dir", "")
    if latest_rows and latest_result_dir:
        _render_rename_and_browser(latest_rows, latest_result_dir)

elif mode == "Mode 1: Model Training & Registration":
    st.header("Mode 1: Training & Setup")
    st.warning("该模式用于新类别注册与训练流程触发（当前为占位页）。")

elif mode == "Mode 2: Single Stream Inference":
    st.header("Mode 2: Single Stream Inference")
    st.info("单张图像进入后立即返回结果（函数式输出），固定阈值 0.5，anomaly_type 默认使用 cluster_id。")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_root = os.path.join(project_root, "models_store")
    model_candidates = _collect_models_recursive(models_root)

    st.caption(f"模型扫描目录：{models_root}")

    selected_model_path = ""
    if model_candidates:
        select_options = ["(不选择)"] + [os.path.relpath(path, project_root) for path in model_candidates]
        selected_option = st.selectbox("从 models_store 选择模型", options=select_options, index=1)
        if selected_option != "(不选择)":
            selected_model_path = os.path.join(project_root, selected_option)
    else:
        st.warning("未在 models_store 下发现可用模型文件（.pt/.pth/.ckpt/.bin）。")

    model_path_override = st.text_input("自定义模型路径（可选，填了会覆盖上面的选择）", value="")
    model_path = model_path_override.strip() or selected_model_path
    st.write(f"当前使用模型路径：{model_path if model_path else '未指定'}")

    single_file = st.file_uploader("上传单张图像", accept_multiple_files=False, type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if st.button("Run Single Image Pipeline"):
        if single_file is None:
            st.error("请先上传一张图像。")
        else:
            single_input_dir = os.path.join(project_root, "data_store", "raw_single")
            if os.path.exists(single_input_dir):
                shutil.rmtree(single_input_dir)
            os.makedirs(single_input_dir, exist_ok=True)

            single_image_path = os.path.join(single_input_dir, single_file.name)
            with open(single_image_path, "wb") as image_file:
                image_file.write(single_file.getbuffer())

            try:
                with st.spinner("Running single-image inference..."):
                    result = _run_single_image_pipeline(single_image_path, project_root, model_path=model_path)
                st.success("单张推理完成。")
                st.json(result)

                preview_col1, preview_col2 = st.columns(2)
                with preview_col1:
                    st.image(Image.open(single_image_path), caption="输入图像")
                with preview_col2:
                    st.metric("异常分数 score", result["score"])
                    st.metric("cluster_id", result["cluster_id"])
                    st.metric("anomaly_type", result["anomaly_type"])
                    st.metric("latency_ms", result["meta"]["latency_ms"])
            except Exception as run_error:
                st.error(f"单张推理失败：{run_error}")
