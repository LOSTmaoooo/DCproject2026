# 复现指南（How to Reproduce）

> 本文档回答三个问题：**仓库里有什么**、**数据和权重怎么准备**、**怎么跑通**。
> 适用于从零开始在一台新机器上复现本项目的两套流程。

---

## 0. 仓库里有什么 / 没有什么（重要）

### ✅ git 仓库**包含**的

| 内容 | 位置 | 说明 |
|------|------|------|
| 核心胶水层 | `core/*.py`（engine / data_bridge / musc_wrapper / AnomalyNCD_wrapper / run_pipeline）| 本工程自己写的，含 6 处 bug 修复 |
| 前端 | `app/main_app.py` | Streamlit 界面 |
| 两个算法库 | `libs/MuSc/`、`libs/AnomalyNCD/` | 论文官方代码 + 关键修复 |
| 运行脚本 | `run_train.sh`、`run_standard_mtd.sh`、`gen_mtd_maps.py` | 一键跑训练/预处理 |
| 文档 | `PROJECT_GUIDE.md`、`PROJECT_ANALYSIS.md`、`MUSC_PAPER_GUIDE.md`、`ANOMALYNCD_PAPER_GUIDE.md`、`RESEARCH_PLAN_DIRECTION1.md`、`REPRODUCTION_REPORT.md` | 上手 / 原理 / 研究计划 |

### ❌ git 仓库**不包含**的（`.gitignore` 已忽略，需单独准备）

| 内容 | 为什么不在仓库里 | 去哪准备 |
|------|----------------|---------|
| **数据集**（AeBAD / mtd / mvtec）| 太大（几个 GB），且是第三方数据 | 见第 2 节 |
| **预训练权重**（CLIP / DINOv2）| 数百 MB~1GB，是第三方模型 | 见第 3 节 |
| 运行时产物（结果、模型 checkpoint、日志）| 机器相关、可再生 | 跑起来自动生成 |

---

## 1. 环境搭建

### 1.1 系统依赖

```bash
apt update && apt install libgl1-mesa-glx libglib2.0-0 -y   # cv2 需要
```

### 1.2 Python 环境

本机实测用的是 **Python 3.10（conda base 环境）**，PyTorch 2.1.2 + CUDA 11.8。代码兼容 3.8~3.10。

```bash
# 1. PyTorch（cu118）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 2. 核心依赖（MuSc 用的是内置 open_clip 副本，不需要单独装 open-clip-torch）
pip install opencv-python==4.9.0.80 timm==0.9.12 streamlit

# 3. 两个算法库的依赖
pip install scikit-learn==1.3.2 scipy==1.10.1 pandas loguru scikit-image ftfy openpyxl
```

> 说明：MuSc 的 `libs/MuSc/models/backbone/open_clip/` 是**内置的 open_clip 副本**（`import models.backbone.open_clip`），所以**不要**另外 `pip install open-clip-torch`，装了的反而要卸载，否则和 `timm 0.9.12` 冲突。

> 国内 pip 源建议加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`（阿里云源实测限速 ~150KB/s 很慢）。

---

## 2. 数据集准备（位置 + 格式）

### 2.1 三个数据集，都放在 `libs/AnomalyNCD/data/` 下

```
libs/AnomalyNCD/data/
├── AeBAD_crop/                    # base 数据集（有标签异常先验）
├── mtd_anomaly_detection/         # novel 数据集（无标签，待发现）
└── mvtec_anomaly_detection/       # 可选，另一套 novel 数据
```

> **关于软链接**：本机（autodl）为省系统盘，把这三个目录做成了指向数据盘 `/root/autodl-tmp/DCproject/datasets/` 的软链接。**复现时不需要软链接**，直接把数据解压到 `libs/AnomalyNCD/data/` 下即可，代码用的是相对路径，不受影响。

### 2.2 各自的内部格式（必须一致）

**AeBAD_crop**（base，4 类缺陷）：
```
AeBAD_crop/
├── images/{ablation, breakdown, fracture, groove}/   # 每类若干张图
└── masks/{ablation, breakdown, fracture, groove}/    # 对应的分割掩码
```

**mtd_anomaly_detection**（novel，磁瓦缺陷）：
```
mtd_anomaly_detection/
├── train/MT_Free/                  # 正常样本（761 张）
├── test/{MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Free, MT_Uneven}/
└── ground_truth/                   # 测试集的分割掩码
```

**mvtec_anomaly_detection**（novel，15 类标准工业缺陷）：
```
mvtec_anomaly_detection/
├── bottle/{train/good, test/{broken_large, ...}, ground_truth/}
├── cable/ ...
└── ...（共 15 类）
```

### 2.3 数据从哪来

| 数据集 | 来源 |
|--------|------|
| AeBAD | 论文官方（GitHub `zhangzilongc/MMR` 提供 AeBAD，`AeBAD_crop` 是官方预处理好的子集）|
| mtd | GitHub `abin24/Magnetic-tile-defect-datasets`（原始 MTD，需按 `libs/AnomalyNCD/datasets/mtd_preprocess.py` 预处理成 train/test/ground_truth）|
| mvtec | MVTec 官网（需注册下载）|

> 如果你手里已经有现成的 zip（如本项目之前的 `AeBAD_crop.zip`、`mtd_anomaly_detection.zip`、`mvtec_anomaly_detection.zip`），直接解压到 `libs/AnomalyNCD/data/` 下对应名字即可。

---

## 3. 预训练权重准备（backbone）

两个 backbone 的权重**在第一次运行时会自动下载**到 `~/.cache/`，也可以手动提前下载（国内网络推荐手动）。

### 3.1 MuSc 的 CLIP 权重

- 文件：`ViT-L-14-336px.pt`（约 891MB）
- 下载 URL：`https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt`
- 放置位置：`~/.cache/clip/ViT-L-14-336px.pt`

```bash
mkdir -p ~/.cache/clip
wget -c -O ~/.cache/clip/ViT-L-14-336px.pt "<上面的 URL>"
```

### 3.2 AnomalyNCD 的 DINO 权重

- 文件：`dino_vitbase8_pretrain.pth`（约 328MB）
- 下载 URL：`https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth`
- 放置位置：`~/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth`

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -c -O ~/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth "<上面的 URL>"
```

> 放到指定缓存路径后，代码里的 `torch.hub.load_state_dict_from_url` / open_clip 会**自动命中缓存、不再联网下载**。

---

## 4. 跑通验证

本项目有**两条流程**，都已验证能收敛（结果超过论文官方）。

### 4.1 流程 A：标准 benchmark（AeBAD base + mtd novel，复现论文）

这是复现论文官方设定的流程，**base 用 AeBAD 的 4 类异常**。

```bash
# 第 1 步：用 MuSc 生成 mtd 的 anomaly maps（全局归一化，约 40 分钟）
cd /root/DCproject2026   # 换成你的项目路径
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
python -u gen_mtd_maps.py

# 第 2 步：跑 AnomalyNCD 训练（约 3 小时，100 epochs）
bash run_standard_mtd.sh
# 或进 tmux 后台跑：tmux new -s train bash run_standard_mtd.sh
```

**预期结果**（日志末尾）：
```
labeled class: ['ablation', 'breakdown', 'fracture', 'groove']
unlabeled class: ['MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
... cls_loss 十几个 epoch 降到 ~0.1，最终 ~0.005
region-merge: NMI ~0.29 | ARI ~0.22 | F1 ~0.50
```

### 4.2 流程 B：core 全流程（`run_pipeline.py`，自定义数据流水线）

这是本工程自己封装的一键流水线，base 同样用 AeBAD（已修复）。

```bash
# 准备输入数据：一个目录，内含 known_normal（正常）+ 各异常类子目录
# 例如 /path/to/input/known_normal/*.jpg + /path/to/input/MT_Blowhole/*.jpg ...

# 一键跑（MuSc → DataBridge → AnomalyNCD）
bash run_train.sh /path/to/input
# 或：python -u core/run_pipeline.py --input_dir /path/to/input
```

**输入数据格式**（`known_normal` 是必须的正常参照，其余目录是待发现的异常类）：
```
input_dir/
├── known_normal/xxx.jpg    # 正常样本（MuSc 用它当参照）
├── MT_Blowhole/xxx.jpg     # 异常类（待发现）
├── MT_Break/xxx.jpg
└── ...
```

### 4.3 运行时数据写到哪（`DATA_ROOT`）

core 流程的上传图、结果、模型默认写到 `DC_DATA_ROOT` 指定的目录（默认 `/root/autodl-tmp/DCproject`，这是 autodl 的数据盘）。**换机器复现时**，设环境变量即可：

```bash
export DC_DATA_ROOT=/你的/数据目录
```

不改环境变量也能跑，只是默认路径是 autodl 特有的。代码里所有运行时路径都走这个变量。

---

## 5. 关键 bug 修复清单（为什么要这个版本）

这个版本相比原始代码，修复了 6 处「换数据/换平台就崩」的 bug，**复现时务必用这个版本**：

| # | 问题 | 修复位置 |
|---|------|---------|
| 1 | base 用错成「正常类」导致不收敛 | `core/data_bridge.py` |
| 2 | lib 被改坏（base_category / normal_ref）| `libs/AnomalyNCD/models/AnomalyNCD.py`、`datasets/data_utils.py` |
| 3 | anomaly map 逐张归一化导致 MEBin 全黑 | `core/data_bridge.py` |
| 4 | MuSc MSM 大矩阵 OOM | `libs/MuSc/models/modules/_MSM.py`（分批 cdist）|
| 5 | Windows→Linux 模块名大小写 | `core/engine.py` |
| 6 | 结果路径层级算错 | `libs/AnomalyNCD/models/AnomalyNCD.py`（`../../../../`→`../../../`）|

---

## 6. 常见问题

| 问题 | 排查 |
|------|------|
| `No module named 'open_clip'` | 别装 open-clip-torch；确认 `libs/MuSc/models/backbone/open_clip/` 存在 |
| MuSc 阶段 OOM | 显存不够：`libs/MuSc/models/modules/_MSM.py` 的 `chunk_size` 调小（如 100→30）|
| AnomalyNCD OOM | `libs/AnomalyNCD/configs/AnomalyNCD.yaml` 的 `batch_size` 调小（32→16）|
| cls_loss 不下降 | 确认 base 用的是 AeBAD（异常类），不是正常类 |
| 权重下载慢/失败 | 按第 3 节手动下载到 `~/.cache` |

---

*本仓库代码以 git 提交历史为准（关键提交：`bdcf3ce fix: 修复 AnomalyNCD 不收敛与 MuSc OOM`、`180e3c2 refactor: 迁移数据盘`），数据集与权重为第三方资源，按上述方式自备。*
