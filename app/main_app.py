# -*- coding: utf-8 -*-
"""FastAPI 后端入口。

职责：
1. 暴露 3 个模式的 API（当前重点是 Mode 3 批处理）。
2. 接收 ZIP 数据并调用 `core.engine.BatchPipeline` 执行异常识别流水线。
3. 统一返回结构，便于前端与外部系统集成。
"""

import os
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 将项目根目录加入模块搜索路径，确保可导入 `core` 等包。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from core.engine import BatchPipeline
except ImportError as import_error:
    print(f"Error importing core engine: {import_error}")
    BatchPipeline = None

try:
    from core.musc_wrapper import MuScWrapper
except ImportError as import_error:
    print(f"Error importing MuSc wrapper: {import_error}")
    MuScWrapper = None


app = FastAPI(
    title="Industrial Anomaly Detection API",
    description="Backend API supporting 3 Operation Modes",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UNKNOWN_THRESHOLD = 0.5
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_EXTS = (".pt", ".pth", ".ckpt", ".bin")
_MUSC_WRAPPER = None


class ResponseModel(BaseModel):
    """统一响应体。"""

    code: int
    msg: str
    data: Optional[Any] = None


def success_response(data: Any = None, msg: str = "请求成功"):
    return {"code": 200, "msg": msg, "data": data}


def error_response(code: int, msg: str):
    return {"code": code, "msg": msg, "data": None}


def _normalize_score(map_path: str) -> float:
    if not os.path.exists(map_path):
        return 0.0
    anomaly_map = np.load(map_path)
    if anomaly_map.size == 0:
        return 0.0
    score = float(anomaly_map.max())
    if score > 1.0:
        score = score / 255.0
    return float(max(0.0, min(1.0, score)))


def _score_to_cluster(score: float) -> int:
    return 1 if score >= UNKNOWN_THRESHOLD else 0


def _get_musc_wrapper(project_root: str):
    global _MUSC_WRAPPER
    if MuScWrapper is None:
        return None
    if _MUSC_WRAPPER is None:
        config_path = os.path.join(project_root, "libs", "MuSc", "configs", "musc.yaml")
        _MUSC_WRAPPER = MuScWrapper(config_path=config_path)
    return _MUSC_WRAPPER


def _resolve_model_path(project_root: str, model_name_or_path: str) -> str:
    if not model_name_or_path:
        return ""
    if os.path.isabs(model_name_or_path):
        return model_name_or_path if os.path.exists(model_name_or_path) else ""

    models_root = os.path.join(project_root, "models_store")
    candidate = os.path.join(models_root, model_name_or_path)
    if os.path.exists(candidate):
        return candidate

    for current_root, _, files in os.walk(models_root):
        for file_name in files:
            if not file_name.lower().endswith(MODEL_EXTS):
                continue
            if file_name == model_name_or_path:
                return os.path.join(current_root, file_name)
    return ""


@app.post("/api/v1/mode1/register", response_model=ResponseModel, tags=["Mode 1: Training"])
async def register_new_part(part_name: str = Form(...), images: UploadFile = File(...)):
    """模式1占位接口：零件注册与训练触发（尚未接入实际训练流程）。"""
    _ = images
    return success_response(msg=f"Mode 1: Registered part '{part_name}'. Training feature is under development.")


@app.post("/api/v1/mode2/predict", response_model=ResponseModel, tags=["Mode 2: Inference"])
async def single_stream_inference(
    image: UploadFile = File(..., description="Single image for online inference"),
    model_name: str = Form(default=""),
):
    """模式2在线接口：单张图像同步推理，拍照后可直接调用。"""
    if MuScWrapper is None:
        return error_response(500, "后端核心模块（MuScWrapper）导入失败")

    image_name = image.filename or "input.png"
    if not image_name.lower().endswith(IMAGE_EXTS):
        return error_response(400, "仅支持图片文件（png/jpg/jpeg/bmp/tiff）")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    request_root = os.path.join(project_root, "data_store", "results", f"online_{request_id}")
    input_dir = os.path.join(request_root, "input")
    maps_dir = os.path.join(request_root, "anomaly_maps")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)

    input_path = os.path.join(input_dir, image_name)
    with open(input_path, "wb") as image_file:
        shutil.copyfileobj(image.file, image_file)

    resolved_model_path = _resolve_model_path(project_root, model_name)

    try:
        wrapper = _get_musc_wrapper(project_root)
        if wrapper is None:
            return error_response(500, "MuScWrapper 初始化失败")

        start_time = time.perf_counter()
        wrapper.generate_anomaly_maps(input_dir, maps_dir)
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        stem = Path(image_name).stem
        map_path = os.path.join(maps_dir, f"{stem}_map.npy")
        score = _normalize_score(map_path)
        cluster_id = _score_to_cluster(score)

        return success_response(
            data={
                "image": image_name,
                "score": round(score, 6),
                "cluster_id": cluster_id,
                "anomaly_type": str(cluster_id),
                "is_unknown": False,
                "paths": {
                    "input": input_path,
                    "anomaly_map": map_path if os.path.exists(map_path) else "",
                },
                "meta": {
                    "threshold": UNKNOWN_THRESHOLD,
                    "latency_ms": latency_ms,
                    "model_name": model_name,
                    "resolved_model_path": resolved_model_path,
                    "request_dir": request_root,
                },
            },
            msg="单张在线推理成功",
        )
    except Exception as run_error:
        traceback.print_exc()
        return error_response(500, f"单张在线推理失败: {run_error}")


@app.post("/api/v1/mode3/batch_analysis", response_model=ResponseModel, tags=["Mode 3: Batch Analysis"])
async def start_batch_analysis(file: UploadFile = File(..., description="ZIP file containing dataset or images")):
    """模式3批处理接口。

    处理流程：
    1. 接收 ZIP 包并解压到临时目录。
    2. 自动定位有效输入目录（兼容单层目录嵌套）。
    3. 调用 `BatchPipeline.run(input_dir)` 执行完整流水线。
    4. 返回结果目录位置。
    """
    if BatchPipeline is None:
        return error_response(500, "后端核心模块（BatchPipeline）导入失败")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_dir = os.path.join(project_root, "data_store", "api_uploads")

    # Demo 场景下每次请求清空目录；生产环境建议改为“每任务唯一目录”。
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

    extract_base_path = os.path.join(upload_dir, "extracted")
    os.makedirs(extract_base_path, exist_ok=True)

    try:
        file_name = file.filename or "uploaded.zip"
        file_path = os.path.join(upload_dir, file_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        input_dir = extract_base_path

        if file_name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_base_path)
            except zipfile.BadZipFile:
                return error_response(400, "无效的 ZIP 文件")
        else:
            return error_response(400, "请上传 ZIP 压缩包")

        subdirs = [
            item
            for item in os.listdir(extract_base_path)
            if os.path.isdir(os.path.join(extract_base_path, item))
        ]

        if len(subdirs) == 1:
            potential_root = os.path.join(extract_base_path, subdirs[0])
            has_images = any(
                name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
                for name in os.listdir(potential_root)
            )
            if has_images or os.path.isdir(potential_root):
                input_dir = potential_root

        print(f"[API] Resolved input directory to: {input_dir}")
        print("[API] Starting batch pipeline...")

        pipeline = BatchPipeline()
        success = pipeline.run(input_dir)

        if success:
            result_root = os.path.join(project_root, "data_store", "results")
            return success_response(
                data={
                    "status": "finished",
                    "result_dir": result_root,
                    "info": "Results are saved in data_store/results on the server.",
                },
                msg="批处理异常识别执行成功",
            )

        return error_response(500, "批处理流水线执行失败（请查看终端日志）")

    except Exception as run_error:
        traceback.print_exc()
        return error_response(500, f"服务器内部错误: {run_error}")


if __name__ == "__main__":
    print("API Server starting...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
