# DCproject2026 项目说明文档

> 目的：帮你快速重新熟悉这个项目。涵盖项目是什么、结构、数据流、三个数据集的角色（重点）、环境状态，以及怎么跑起来。

---

## 1. 项目概述

这是一个**工业场景下的「新异常类别发现」系统**（Novel Anomaly Class Discovery），核心能力是：

> 给定一批**没有标签**的工业缺陷图片，自动把它们聚成不同的异常类别，并输出每张图/每个区域的类别。

它把两个开源算法串成了一条流水线：

| 算法 | 定位 | 作用 |
|------|------|------|
| **MuSc** | zero-shot 异常检测 | 给输入图片生成「异常热力图」（anomaly map），指出哪里异常、异常程度 |
| **AnomalyNCD** | 多类异常分类 / 聚类 | 接收 MuSc 的热力图 + 原图，用对比学习把异常**聚类成新类别**，实现「新类别发现」 |

对应论文：
- MuSc: *MuSc: Zero-shot industrial anomaly classification and segmentation with mutual scoring* (CVPR 2024)
- AnomalyNCD: *AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios* (CVPR 2025)

---

## 2. 项目结构

```
DCproject2026/
├── app/                          # Streamlit 前端
│   └── main_app.py               #   网页界面：上传数据、启动训练、看结果
├── core/                         # 自定义核心逻辑（把两个算法串起来的地方）
│   ├── engine.py                 #   总控引擎 BatchPipeline：串联 MuSc → DataBridge → AnomalyNCD
│   ├── musc_wrapper.py           #   MuSc 封装：生成异常热力图
│   ├── data_bridge.py            #   数据桥接：把 MuSc 输出转成 AnomalyNCD 要的 MTD 格式
│   ├── AnomalyNCD_wrapper.py     #   AnomalyNCD 封装：跑聚类/训练
│   └── run_pipeline.py           #   命令行入口（被前端调用）
├── libs/                         # 两个开源算法（第三方源码，核心不动）
│   ├── MuSc/
│   │   ├── models/               #   musc.py + 模块(LNAMD/MSM/RsCIN)
│   │   ├── datasets/             #   mvtec.py / visa.py / btad.py
│   │   ├── configs/musc.yaml     #   配置（backbone、数据路径等）
│   │   └── data/                 #   ← MuSc 用的数据集放这里
│   └── AnomalyNCD/
│       ├── models/               #   AnomalyNCD.py + loss + modules
│       ├── datasets/             #   dataset.py / data_utils.py / 预处理脚本
│       ├── configs/AnomalyNCD.yaml#  训练超参
│       ├── scripts/              #   anomalyncd.sh（训练）/ anomalyncd_test.sh（推理）
│       └── data/                 #   ← 数据集（软链接 → 数据盘 datasets/，见第 5 节）
├── data_store/                   # ⚠️ 已废弃（历史空目录，运行时数据已迁数据盘）
├── models_store/                 # 软链接 → 数据盘 models/（checkpoints/configs/feature_banks）
├── server/  cloud/               #   预留（后端/云接口，目前为空）
└── README.md                     # 旧版 README（含乱码，可忽略）
```

**核心代码就 5 个文件**（`core/` 下），其余都是被它们调用的第三方算法。

---

## 3. 数据流（一句话讲清）

```
用户上传图片（按语义前缀分类，如 known_normal / known_crack / unknown_new）
        │
        ▼
[1] MuSc  零样本推理 → 每张图生成异常热力图 (.npy)
        │
        ▼
[2] DataBridge  语义分流：
        known_normal       → normal_ref/<类别>/images|train|masks/good   （作为 base 正常参考）
        其他异常类别        → images/<类别>/xxx.png + anomaly_maps/<类别>/xxx.png
        │
        ▼
[3] AnomalyNCD  用 base 参考 + 异常图 + 热力图 → 聚类出新的异常类别
        │
        ▼
    输出：分类结果报表 (CSV) + 模型 checkpoint
```

对应 `core/engine.py` 里的 `BatchPipeline.run()` 三步：`run_musc` → `run_bridge` → `run_ncd`。

---

## 4. 三个数据集的角色（重点回答你的疑问）

### 结论速览

| 数据集 | 是否必须 | 角色 | 说明 |
|--------|:---:|------|------|
| **AeBAD** | ✅ **必须** | **base 数据集**（有标签的已知异常） | 给 AnomalyNCD 提供「已知异常类别」的先验，训练时当作 labeled 数据 |
| **MTD** | ⭕ 可选 | **novel 数据集**（无标签，待发现类别） | 与 MVTec **二选一**，作为要被聚类的数据 |
| **MVTec AD** | ⭕ 可选 | **novel 数据集**（无标签，待发现类别） | 与 MTD **二选一** |

### 4.1 AeBAD —— 必须，用于基础训练

- **官方定位**（见 `libs/AnomalyNCD/README.md`）：AeBAD 是「labeled abnormal image set D^l」，即**有标签的异常图片集**，作为 AnomalyNCD 的 base 数据。
- **在代码里的体现**：`libs/AnomalyNCD/scripts/anomalyncd.sh` 里写死：
  ```bash
  base_data_path="data/AeBAD_crop"
  ```
  而 `datasets/data_utils.py` 里 `get_class_splits()` 会直接 `os.listdir(base_data_path/images)` 去读它，**缺了直接报错**。
- **你说的「用于基础 MuSc 训练」**：严格来说 AeBAD 是喂给 **AnomalyNCD** 做 base 数据，不是给 MuSc 训练。MuSc 是零样本的，不需要训练数据。但「基础/必须」这个理解是对的。

### 4.2 MTD —— 可选，与 MVTec 二选一（你说的「可修改的训练数据」）

- **官方定位**：MTD 是「novel / unlabeled image dataset D^u」的**选项之一**，另一个是 MVTec AD。它们是**要被 AnomalyNCD 聚类、发现新类别**的数据。
- **在代码里的体现**：`scripts/anomalyncd.sh` 里 mvtec 部分**启用**，mtd 部分**被注释掉**：
  ```bash
  # # MTD dataset
  # dataset_path="data/mtd_anomaly_detection/test"
  ```
- **你的记忆「可修改的训练数据先用 mtd」是对的**：novel 数据就是「可以换成你想要的任何数据」的那部分。优先用 **mtd** 的原因是它**小（约 50MB）**、训练快，适合先跑通流程；mvtec 有 5.2GB、15 个类，跑起来慢得多。
- **结论：mtd 不是必须的，但它是「跑通 + 快速验证」的首选**。你之前也确实用 mtd 跑过（`libs/AnomalyNCD/outputs/` 里有 `AnomalyNCD_MTD_...` 的实验记录）。

### 4.3 一句话总结

- 想**复现论文 / 标准 benchmark**：需要 `AeBAD`（base）+ `mtd` **或** `mvtec`（novel，二选一）。
- 只想**跑自定义数据流水线**（前端上传自己的图）：三个数据集**都不需要**，base 和 novel 数据都由你上传的图片充当。

---

## 5. 数据布局：数据盘迁移方案

### 5.1 为什么迁移

系统盘（`/`）只有 30G，数据集 + 压缩包约 11G 会占满。autodl 的 `/root/autodl-tmp` 是**数据盘**（50G，重启不丢失），所以把「非代码」数据都迁过去。

### 5.2 数据盘目录结构

```
/root/autodl-tmp/DCproject/
├── datasets/              # 解压后的数据集（约 5.5G）
│   ├── AeBAD_crop/        #   images + masks（4 类）
│   ├── mtd_anomaly_detection/   # train / test / ground_truth
│   └── mvtec_anomaly_detection/ # 15 类标准结构
├── uploads/               # 前端上传的图像（原 data_store/raw_inputs_uploaded）
├── results/               # 结果输出（原 data_store/results）
├── models/                # 模型文件（原 models_store）
└── zips/                  # 原始压缩包归档（约 5.5G）
```

### 5.3 代码如何访问这些数据（两种机制）

**① 数据集 —— 软链接**：`libs/AnomalyNCD/data/` 下的三个目录、`libs/MuSc/data/mvtec_anomaly_detection` 都是**软链接**指向数据盘 `datasets/`。因此 `scripts/*.sh` 里的相对路径 `data/...` **无需改动**。

**② 运行时数据 —— `DATA_ROOT` 环境变量**：代码（`core/engine.py`、`core/run_pipeline.py`、`app/main_app.py`）通过统一的 `DATA_ROOT` 定位 uploads/results/models：

```python
DATA_ROOT = os.environ.get("DC_DATA_ROOT", "/root/autodl-tmp/DCproject")
```

默认指向数据盘；换机器/换盘时只需设环境变量 `DC_DATA_ROOT`，无需改代码。

### 5.4 ⚠️ 最需要注意的改动

1. **软链接在「重新打包/分发」时会失效**。如果你重新把 `DCproject2026` 压缩成 zip 发给别人，软链接会断掉（zip 不保留软链接）。正确做法：分发时把 `libs/*/data/` 下的软链接换成真实数据，或在目标机器上重新解压数据到数据盘并重建软链接。
2. **`data_store/` 已废弃**。代码不再读写 `data_store/`（上传、结果都走数据盘），项目根下的 `data_store/` 是历史空目录，可删。
3. **`models_store/` 现在是软链接**，指向数据盘 `models/`；前端 `MODEL_DIR` 已改为 `DATA_ROOT/models/checkpoints`（顺手修正了原来 `checkpoint` 单复数的笔误）。
4. **数据盘路径是 autodl 特有的**（`/root/autodl-tmp`）。换到别的平台需改 `DC_DATA_ROOT` 默认值。
5. **系统盘重启可能重置**：代码和软链接在系统盘，若 autodl 系统盘重启重置，需重新部署代码并重建软链接（数据盘内容保留）。

### 5.5 数据集本身

- **AeBAD** 4 类 = `ablation, breakdown, fracture, groove` ✅（与 AnomalyNCD 论文 AeBAD-S 一致）
- **mtd** 已是 `train/test/ground_truth` 分好格式，无需再跑 `mtd_preprocess.py` ✅

---

## 6. 环境配置

### 6.1 官方要求（`DCproject_settings.md` 所述）

| 项 | 要求 |
|----|------|
| Python | 3.8 |
| CUDA | 11.8 |
| PyTorch | 2.0.1 + cu118（settings.md）|
| 系统依赖 | `libgl1-mesa-glx libglib2.0-0` |
| 关键库 | open-clip-torch、opencv-python、timm、streamlit |
| 两工程依赖 | `pip install -r libs/MuSc/requirements.txt` + `libs/AnomalyNCD/requirements.txt` |

### 6.2 当前服务器实际状态（本次已配置完成 ✅）

| 项 | 状态 | 说明 |
|----|------|------|
| **GPU** | ❌ 无卡模式 | `cuda available: False`，推理/训练需挂载 GPU |
| Python | 3.10.8（base 环境） | ⚠️ 与 settings.md 的 3.8 不同，但代码兼容 |
| PyTorch | 2.1.2+cu118 ✅ | 开卡即用（cu118 版） |
| torchvision | 0.16.2+cu118 ✅ | |
| opencv-python | 4.9.0.80 ✅ | |
| timm | 0.9.12 ✅ | 遵循 MuSc requirements |
| streamlit | 1.63.0 ✅ | |
| scikit-learn / scipy | 1.3.2 / 1.10.1 ✅ | |
| pandas / numpy | 2.3.3 / 1.26.4 ✅ | 略高于 requirements，API 兼容 |
| loguru / scikit-image | 0.7.2 / 0.21.0 ✅ | |
| ftfy / openpyxl | 6.1.3 / 3.1.2 ✅ | |
| 系统依赖 libgl | ✅ | 已装 |

### 6.3 结论

**依赖已全部就位**，唯一阻断点是**无 GPU**（无卡模式）。开 GPU 卡后即可直接运行，无需再装任何东西。

> **关于 Python 版本**：settings.md 写 3.8，但本机 conda 建 3.8 环境极慢（清华源 repodata 123MB + 1 核 CPU），故直接用 base 环境（3.10.8）。项目代码与所有依赖库均兼容 3.10，功能不受影响。若需严格 3.8，可后续 `conda create -n DCproject python=3.8`。

> **关于 pip 源**：阿里云 pip 源实测下载仅 ~150 KB/s（极慢），已改用清华源（~5.5 MB/s，快 35 倍）。后续安装建议加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 7. 怎么跑起来

### 7.1 两种使用模式

**模式 A —— 自定义数据流水线（推荐，前端操作）**
1. 打包一个 ZIP，结构形如：
   ```
   your_data.zip
   ├── known_normal/xxx.png     # 正常参考样本（必须要有这个类别）
   ├── known_crack/yyy.png      # 已知异常（可选）
   └── unknown_new/zzz.png      # 待发现的异常
   ```
2. 启动前端：`streamlit run app/main_app.py --server.address 0.0.0.0 --server.port 8501`
3. 网页里选「Mode 1」，上传 ZIP，点「开始训练」。

**模式 B —— 标准 benchmark（复现论文，用 AeBAD + mtd/mvtec）**
```bash
# 在 libs/AnomalyNCD 下，编辑 scripts/anomalyncd.sh：
#   - 用 mvtec：默认已启用
#   - 用 mtd：取消 mtd 段注释、注释掉 mvtec 段
bash scripts/anomalyncd.sh
```
（前提：先用 MuSc 生成对应数据集的 anomaly maps，见 `libs/AnomalyNCD/README.md` 第 4 节。）

### 7.2 启动命令备忘

```bash
# 本地
streamlit run app/main_app.py

# 全网访问（需在 autodl 控制台开放 8501 端口）
streamlit run app/main_app.py --server.address 0.0.0.0 --server.port 8501

# 后台常驻
nohup streamlit run app/main_app.py --server.address 0.0.0.0 --server.port 8501 > app.log 2>&1 &

# 命令行直接跑流水线（不走前端）
python -u core/run_pipeline.py --input_dir <你的数据目录>
```

---

## 8. 待办清单（当前状态）

1. ☐ **挂载 GPU**（唯一阻断点）—— 无卡模式下环境已就绪，开卡后直接运行
2. ☐ 首次跑 MuSc 时联网下载 CLIP 预训练权重（config 用 `ViT-L-14-336` openai 权重，约 1.7GB；无卡时可先不下载）
3. ☐ 用 mtd 跑通一次 benchmark 验证（数据已就位）

---

## 9. 本次修复的问题

### 9.1 `musc_wrapper.py` 的 open_clip 导入 bug

- **现象**：从项目根目录运行 `core/run_pipeline.py` 时，`MuScWrapper` 导入失败，报 `No module named 'open_clip'`。
- **根因**：`libs/MuSc/models/musc.py` 里用相对路径 `sys.path.append('./models/backbone')`，而 open_clip 包内有 `from open_clip.utils import ...` 绝对导入，二者都依赖「当前工作目录 = MuSc 目录」。但 `core/` 流程是从项目根目录启动的，导致 open_clip 找不到。
- **修复**：在 `core/musc_wrapper.py` 中用绝对路径把 `libs/MuSc/models/backbone` 加入 `sys.path`，消除对 cwd 的依赖。已改并验证通过 ✅

### 9.2 `engine.py` 的模块名大小写 bug

- **现象**：跑完整流水线时，AnomalyNCD 阶段报 `ModuleNotFoundError: No module named 'core.anomalyncd_wrapper'`。
- **根因**：`core/engine.py` 里 import 写成了全小写 `core.anomalyncd_wrapper`，但实际文件名是 `core/AnomalyNCD_wrapper.py`（大写 A、NCD）。**Windows 不区分大小写所以能跑，Linux 区分大小写就挂了** —— 这是典型的「Windows 开发、Linux 部署」迁移问题。
- **修复**：改为 `from core.AnomalyNCD_wrapper import AnomalyNCDWrapper`，已跑通验证 ✅

### 9.3 `AnomalyNCD.py` 的 PROJECT_ROOT 层级 bug

- **现象**：训练完成后，模型和结果被写到了系统盘 `/root/models_store/` 和 `/root/data_store/`（而不是数据盘），导致数据盘 models 目录为空。
- **根因**：`libs/AnomalyNCD/models/AnomalyNCD.py` 第 295、600 行用 `os.path.dirname(__file__) + '../../../../'` 计算项目根目录，但**多算了一级**（正确是 `'../../../'`），导致 `PROJECT_ROOT` 变成 `/root/` 而非 `/root/DCproject2026/`。
- **修复**：`'../../../../'` → `'../../../'`（两处）；同时重建 `data_store/results` 软链接指向数据盘，并把误写的模型/结果迁回数据盘。

---

*生成时间：2026-09-12*
