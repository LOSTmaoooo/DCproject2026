# -*- coding: utf-8 -*-
# 文件名：core/run_pipeline.py
import argparse
import os
import sys

# 规避底层 C++ (cv2, faiss, scikit-learn 等) 和 PyTorch 多线程打架引发 Segmentation Fault
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 确保能正确导入 core 目录下的包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from core.engine import BatchPipeline

def main():
    parser = argparse.ArgumentParser(description="运行后台分析与训练流水线")
    parser.add_argument("--input_dir", type=str, required=True, help="待处理的输入图片目录")
    args = parser.parse_args()

    print(f"[RunPipeline] 接收到前端指令，准备处理目录: {args.input_dir}")

    # 执行核心流水线
    pipeline = BatchPipeline()
    # 强制将输出目录定为前端监听的 results 目录
    output_dir = os.path.join(PROJECT_ROOT, 'data_store', 'results')
    
    success = pipeline.run(args.input_dir, output_dir=output_dir)

    if success:
        print("[RunPipeline] 全部流程圆满结束！模型文件与分类报表已由底层算法自动输出。")
    else:
        print("[RunPipeline] 运行失败，请向上检查日志报错信息。")
        sys.exit(1)

if __name__ == "__main__":
    main()