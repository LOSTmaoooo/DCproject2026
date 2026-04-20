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
import traceback
import zipfile
from typing import Any, Optional

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


class ResponseModel(BaseModel):
    """统一响应体。"""

    code: int
    msg: str
    data: Optional[Any] = None


def success_response(data: Any = None, msg: str = "请求成功"):
    return {"code": 200, "msg": msg, "data": data}


def error_response(code: int, msg: str):
    return {"code": code, "msg": msg, "data": None}


@app.post("/api/v1/mode1/register", response_model=ResponseModel, tags=["Mode 1: Training"])
async def register_new_part(part_name: str = Form(...), images: UploadFile = File(...)):
    """模式1占位接口：零件注册与训练触发（尚未接入实际训练流程）。"""
    _ = images
    return success_response(msg=f"Mode 1: Registered part '{part_name}'. Training feature is under development.")


@app.post("/api/v1/mode2/predict", response_model=ResponseModel, tags=["Mode 2: Inference"])
async def single_stream_inference(image: UploadFile = File(...)):
    """模式2占位接口：单图实时推理（当前返回示例结果）。"""
    _ = image
    return success_response(
        data={"anomaly_score": 0.0, "is_anomaly": False},
        msg="Mode 2: Single inference feature is under development.",
    )


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
