# 项目技术深度剖析：MuSc + AnomalyNCD + 工程胶水层

> 本文档面向「重新理解这个项目、并思考如何提升它」的场景，分五部分：
> 0. 先厘清核心：为什么要 MuSc + AnomalyNCD 两个算法
> 1. MuSc 算法原理（照论文 + 代码）
> 2. AnomalyNCD 算法原理（照论文 + 代码）
> 3. 本工程除了 lib 之外做了什么（core 胶水层），以及锐评
> 4. 发散性地探索「速度 / 准度」的提升方向

---

## 0. 先厘清核心：为什么要 MuSc + AnomalyNCD 两个算法

很多人（包括写本项目 core 胶水层的那位）都在这里没想明白。两个论文里都叫 "classification"，但**根本不是一回事**：

| | MuSc 的「分类」| AnomalyNCD 的「分类」|
|---|---|---|
| 本质 | 异常检测的**二分类**（正常 vs 异常）| **多类别 + 新类别发现**（裂纹？气孔？刮伤？）|
| 需要预定义类别名 | 要（或只要「正常/异常」二元）| **不要**，自动聚出来 |
| 能发现「没见过的缺陷」 | ❌ 不能 | ✅ 这正是它的核心 |

**一句话分工**：MuSc 回答「有没有异常、异常在哪」；AnomalyNCD 回答「这个异常是哪种缺陷」，而且能**发现从没标注过的新缺陷种类**。

**为什么 MuSc 替代不了 AnomalyNCD**：MuSc 的 zero-shot 分类靠 CLIP 图文对齐，需要给文本 prompt（"裂纹"、"气孔"…）把图归到**已知类别**。但工业缺陷种类会不断冒新，新缺陷没有对应 prompt 就无法正确归类；而且「气孔 vs 裂纹 vs 崩边」这类细微工业语义，CLIP 未必分得清（这跟它对磁瓦 anomaly map 失效是同一个根因——细微语义太弱）。AnomalyNCD 则不需要任何类别名，拿着 base 的异常先验自动把 novel 集聚成簇，**自己发现**「原来有几种不同的缺陷」。

**两者是上下游，不是冗余**：MuSc 输出 anomaly map（异常定位），AnomalyNCD 拿它做 MEBin 二值化提取异常区域 → 聚类归类。AnomalyNCD 论文明确说「compatible with different AD methods」，故意把「异常定位」外包给 MuSc/PatchCore 等，自己只做「分类 + 新类发现」。而且 **MuSc 一作 Xurui Li 就是 AnomalyNCD 的作者之一**，这个串联是作者自己设计的，不是本项目硬凑的。

**什么时候才真正需要 AnomalyNCD**：
- 场景缺陷种类**固定且已知** → 用 MuSc 做分类就够了，AnomalyNCD 确实多余。
- 场景缺陷**动态增长、想自动发现新类型** → 必须上 AnomalyNCD（这正是本项目 "**Novel** Anomaly Class Discovery" 里 Novel 的含义）。

---

## 1. MuSc —— 零样本异常检测与分割

### 1.1 论文定位

**MuSc: Zero-shot Industrial Anomaly Classification and Segmentation with Mutual Scoring**（CVPR 2024，华中科技大学 / 周瑜团队）。

它要解决的问题：工业异常检测通常需要**每个类别单独训练**一个模型（如 PatchCore、PaDiM 都需要在正常样本上建库/训练），成本高、难扩展。MuSc 提出一种 **zero-shot** 方法——**完全不需要训练**，直接用预训练视觉模型（CLIP / DINOv2）的特征，就能同时完成：

- **异常检测**（image-level：这张图有没有异常）
- **异常分割**（pixel-level：异常在哪）

这在工业场景很有价值：新产线、新产品上线时，不用等收集一批正常样本去训练，拿来就能用。

### 1.2 核心思想：Mutual Scoring（互评分）

MuSc 的洞察是：**异常不是「这张图自己长什么样」决定的，而是「这张图和别的图有多不一样」决定的**。

因此它不做「单张图 → 异常分数」的绝对判断，而是做 **mutual scoring（互评分）**：把一批图放在一起，**每张图用其他所有图作为参照**，计算它的每个 patch 在「图群」里的相对异常度。

- 一张**正常**图：它的 patch 和其他正常图的 patch 很像 → 距离小 → 异常分数低
- 一张**异常**图：它的异常 patch 和其他图的 patch 都不像 → 距离大 → 异常分数高

这就是「互」的含义——分数是相对整个图群算出来的，不是绝对阈值。

### 1.3 方法三步

代码在 `libs/MuSc/models/`，主类 `MuSc`（`models/musc.py`），核心模块在 `models/modules/`：

```
输入一批图
   │
   ▼
[Backbone] CLIP(ViT-L-14) 或 DINOv2 提取多层 patch 特征
   │   (取多个中间层，如 feature_layers=[5,11,17,23])
   ▼
[LNAMD] 局部邻域聚合（_LNAMD.py）
   │   把每个 patch 和它周围半径 r 内的邻居 patch 特征聚合，
   │   得到更鲁棒的局部描述子（减少单 patch 噪声）
   ▼
[MSM] 多尺度互评分（_MSM.py）
   │   对每张图 i，计算它的每个 patch 与「其他所有图」的 patch 的最小距离，
   │   取 top-k 最小距离的平均作为该 patch 的异常分数
   ▼
[可选 RsCIN] 分数后处理（_RsCIN.py）
   │   用图级 CLS token 的相似度矩阵做分数传播/平滑
   ▼
异常热力图（anomaly map）
```

**关键细节**（对应代码）：

1. **多层特征融合**：不是只用最后一层，而是取 backbone 的多个中间层（`feature_layers`），每层分别算分数再平均，兼顾低层纹理和高层语义。

2. **LNAMD（Local Neighborhood Aggregation via MD）**：`_LNAMD.py` 里的 `PatchMaker` 先把特征图展开成 patch 序列，然后对每个 patch 聚合其局部邻域（半径 `r`，代码里 `r_list=[1,3,5]` 多尺度）。这一步让每个 patch 的描述子融合了上下文，抗噪更强。

3. **MSM（Multi-Scale / Mutual Scoring）**：`_MSM.py` 的 `compute_scores_fast` 核心是一句 `torch.cdist(Z[i], Z_ref)`——计算图 i 的每个 patch 与「除 i 外所有图的所有 patch」的**欧氏距离**（特征已 L2 归一化，所以距离范围约 [0,2]），然后 `topk(..., largest=False)` 取**最小**的 k 个距离再平均。距离越小 = 越正常，越大 = 越异常。这就是「互评分」的数学实现。

4. **RsCIN（Residual Correction via INterpolation?）**：一个可选后处理，`_RsCIN.py` 里用 `MMO`（基于相似度矩阵 W 做多次近邻传播）把分数在相似图之间「扩散」一次，让分数更平滑。

### 1.4 为什么对 mtd（磁瓦）失效（本项目实测结论）

本文档的调试过程发现了一个关键现象：

| 数据集 | 正常图异常分数(mean) | 异常图异常分数(mean) | 区分度 |
|--------|:---:|:---:|:---:|
| mvtec bottle | 0.232 | 0.329 | ✅ 明显 |
| mtd（磁瓦）| 0.671 | 0.676 | ❌ 几乎重叠 |

**原因**：MuSc 的判别力完全来自预训练 backbone 的特征空间。CLIP 是图文对比学习的，DINO 是自监督的，它们对**纹理、结构、颜色突变**敏感（mvtec 的划痕、破损、污染正是这类），但对**磁瓦这种「对比度极低、语义极弱」的细微缺陷**（表面细微裂纹、针孔气孔）不敏感——正常图和异常图在 backbone 特征空间里几乎分不开，互评分自然失效。

这是 zero-shot 方法的固有边界，不是代码 bug。要解决 mtd，要么换更强的特征提取器（DINOv2-large），要么放弃 zero-shot、改用「需要正常样本建库」的方法（PatchCore/EfficientAD）。

---

## 2. AnomalyNCD —— 新异常类别发现

### 2.1 论文定位

**AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios**（CVPR 2025，华中科技大学）。

它解决的是一个更进阶的问题：**多类别异常分类 + 新类别发现（Novel Class Discovery, NCD）**。

背景：工业场景里，缺陷是**会不断出现新种类**的（今天发现裂纹，明天冒出气孔，后天又来个刮伤）。传统方法要么「每个缺陷类都标注好」做监督分类（太贵），要么「无监督聚类」（缺先验、聚不准）。

AnomalyNCD 的设定（半监督式 NCD）：

- **base 集 D^l（labeled，有标签）**：已知的异常类别，如 AeBAD 的 4 类缺陷（ablation/breakdown/fracture/groove）
- **novel 集 D^u（unlabeled，无标签）**：待发现的新异常类别，如 mtd 的 5 类缺陷

目标：用 base 的「异常先验」去帮 novel 集**自动聚类、发现新类别**，做到「不用标注就能把新缺陷分好类」。

### 2.2 核心难点与创新

论文指出两大难点：

1. **non-prominent anomalies（异常不显著）**：缺陷在图上很不起眼，直接聚类容易被背景噪声带偏。
2. **weak-semantics anomalies（异常语义弱）**：同一种缺陷（如裂纹）在不同图里外观差异大，语义难学。

对应三个创新模块：

1. **MEBin（Main Element Binarization，主元素二值化）**：先把异常检测器（如 MuSc）输出的 anomaly map 二值化，提取「主要异常元素」，裁剪出 **anomaly-centered 子图**——把注意力从整图聚焦到异常区域本身。
2. **Mask-Guided Representation Learning（掩码引导表征学习）**：用 mask 引导网络「只看异常区域」，并用伪标签纠正机制减少错误输入的影响。
3. **Region Merging（区域合并策略）**：先在 region（子图）级别分类，再把属于同一张图的多个 region 合并，得出 image 级别的类别。

### 2.3 方法拆解（对应代码）

主类 `models/AnomalyNCD.py`，入口 `main()` 分两段：`binarization()`（数据预处理）+ `train_init()`/训练循环。

#### (1) MEBin 二值化 + 裁剪（`binarization()`，`models/modules/_MEBin.py`）

```
anomaly map（异常热力图）
   │
   ▼
自适应阈值二值化：对每张 map，从高到低扫描阈值，
  统计不同阈值下的连通域数量，找「稳定区间」确定最佳阈值
   │
   ▼
腐蚀去噪 → 得到 binary mask（异常区域=白）
   │
   ▼
crop：按 mask 的连通域框出子图（sub-image + sub-mask）
```

**这一步是整个流程的命门**：如果 anomaly map 质量差（像 mtd 那样正常/异常分数重叠），MEBin 二值化出来的「异常区域」就是噪声，裁剪出的子图没有区分度，后续聚类必然失败。

#### (2) MGViT 骨干（`load_backbone()`，`models/modules/_MGViT.py`）

Mask-Guided Vision Transformer：在标准 ViT 基础上，把 anomaly mask 作为额外输入，**引导注意力聚焦异常区域**。代码里用 DINO 预训练权重（`dino_vitb8`）初始化，只微调 `grad_from_block` 之后的层。

#### (3) 训练：MGRL（`MGRL()`）

把 base（labeled）+ novel（unlabeled）混合训练，loss 由四部分组成：

| loss | 作用 |
|------|------|
| `cls_loss` | 分类损失（SimGCD 风格，base 的监督 + novel 的伪标签）|
| `cluster_loss` | 聚类损失（distill loss + me_max 熵正则，防止坍缩）|
| `sup_con_loss` | 监督对比损失（base 类内聚、类间散）|
| `contrastive_loss` | 无监督对比损失（info_nce，两视图互增强）|

**收敛的关键信号就是 `cls_loss`**：正常收敛时十几个 epoch 就从 ~1.5 降到 ~0.1，最终到 ~0.005（学长实测）。

#### (4) 推理：sub_image_predict + region_merge_predict

- `sub_image_predict`：对裁剪出的每个子图分类（region 级），输出 NMI/ARI/F1
- `region_merge_predict`：把同图多个 region 的类别合并成 image 级类别

### 2.4 本项目踩过的坑（对理解 AnomalyNCD 很重要）

1. **base 必须是「异常类」**：AnomalyNCD 的 base（labeled）和 novel（unlabeled）**都是异常类**。本项目 core 层一度把 base 换成「正常类」，导致模型没有异常先验，cls 停滞在 0.4 不收敛。改成 AeBAD（4 类异常）做 base 后，cls 立刻收敛到 0.005。
2. **MEBin 依赖 anomaly map 质量**：mtd 的 anomaly map 无区分度 → MEBin 二值化全黑 → 裁剪 fallback 整图 → 不收敛。这是「不收敛」的表层原因，根因在 MuSc 对 mtd 失效。

---

## 3. 本工程除了 lib 做了什么 + 锐评

### 3.1 core 胶水层的架构

`lib/` 下是两个论文的官方代码（基本没动）。工程自己写的是 `core/`（5 个文件）+ `app/`（前端）：

```
core/
├── engine.py               # BatchPipeline：总控，串起 MuSc → DataBridge → AnomalyNCD
├── musc_wrapper.py         # 封装 MuSc：调 generate_anomaly_maps 生成热力图
├── data_bridge.py          # 数据桥接：把用户上传的目录转成 AnomalyNCD 要的格式
├── AnomalyNCD_wrapper.py   # 封装 AnomalyNCD：传参 + 跑训练
└── run_pipeline.py         # 命令行入口（被前端调用）
app/
└── main_app.py             # Streamlit 前端：上传 zip、启动训练、看结果、管模型
```

数据流（core 全流程）：

```
用户上传结构化 zip（known_normal 正常 + 各类异常）
        │
        ▼
[1] MuSc     零样本推理 → 每张图生成 anomaly map (.npy)
        │
        ▼
[2] DataBridge  路由：异常类 → novel；base → 固定用 AeBAD（4 类异常）
        │       （同时做 .npy → 全局归一化 PNG 的格式转换）
        ▼
[3] AnomalyNCD  base(labeled 异常) + novel(未标注异常) → 聚类发现新类别
        │
        ▼
输出：分类报表 CSV + 模型 checkpoint
```

### 3.2 锐评

**做得还行的地方：**

- **模块化封装**：用 wrapper 把两个论文代码包成统一接口，`BatchPipeline` 三段式清晰，思路是对的。
- **动态路径注入**：用 `sys.path` 动态加载 lib，解决了两个 lib 里同名 `models`/`utils` 模块的冲突，是必要的工程技巧。
- **显存管理意识**：MuSc 跑完主动 `del` + `torch.cuda.empty_cache()` 释放显存再跑 AnomalyNCD，这个细节是对的。
- **前端闭环**：Streamlit 上传 → 后台训练 → 结果表格可编辑下载，产品形态完整。

**但问题很严重，而且都是「看起来能跑、换个数据/平台就崩」的隐性 bug：**

1. **根本性设计错误：base 用了「正常类」**（`data_bridge.py`）。这是最致命的——直接违背了 AnomalyNCD「base 和 novel 都是异常类」的核心设定，导致 mtd 上 cls 死活不收敛。属于「没读懂下游算法就写胶水」。

2. **硬编码破坏可移植性**：
   - `AnomalyNCD.py` 里 `base_category = 'good'` 硬编码，`get_pseudo_label_weights` 用 `"normal_ref"` 字符串匹配路径——这些是 core 流程的私货，把 lib 的通用性改没了，标准 benchmark 流程直接 KeyError。
   - `'../../../../'` 路径层级算错，结果/模型写到了错误目录。

3. **归一化 bug**：`data_bridge.py` 原来对每张 anomaly map **逐张 min-max 归一化**，把每张图的 max 都拉成 255，导致 MEBin 的阈值搜索范围退化为 0、二值化全黑。这是「不收敛」的第二个帮凶。

4. **Windows→Linux 迁移 bug**：模块名大小写（`core.anomalyncd_wrapper` vs `AnomalyNCD_wrapper`），Windows 不区分大小写能跑，Linux 直接 `ModuleNotFoundError`。

5. **没考虑显存**：`batch_size=32` + n_views=2 + base 数据量大时 backward OOM；MuSc 的 MSM 用一次性 `torch.cdist` 大矩阵，1153 张图直接爆 12GB 显存。

6. **缺乏文档与测试**：README 是乱码，`DCproject_settings.md` 是过时的手写笔记，没有一条自动化测试，所有 bug 都靠人工跑才发现。

**一句话锐评**：这个工程的「算法选型」是对的（MuSc 零样本打底 + AnomalyNCD 做新类发现，是工业异常检测里一个合理且前沿的组合），但「胶水层」写得太随意——没有读懂下游算法的数据契约，靠硬编码和字符串匹配糊，导致它只在某个特定数据、特定平台下「看起来能跑」，一换场景就系统性崩溃。**典型的「demo 能跑、生产必崩」**。

---

## 4. 发散性探索：如何提升性能

### 4.1 速度

| 方向 | 具体做法 | 预期收益 |
|------|---------|---------|
| **MuSc 特征缓存** | 同一批图，backbone 特征只提取一次，换 r/阈值重算分数时复用；甚至把特征存盘（`.npy`），下次跑直接用 | 省掉最耗时的 backbone forward |
| **MSM 向量检索化** | `torch.cdist` 全对全距离是 O(N²·P²)，可换 **faiss / hnswlib** 做近似最近邻 | 大图群（>万张）时数量级加速 |
| **MSM 显存友好 + 并行** | 现在已改「分批 cdist」；可进一步按图分片多 GPU 并行算分数 | 显存降、吞吐升 |
| **减小 backbone 分辨率** | image_size 518→256，patch 数降 ~4 倍，MSM 距离矩阵降 ~16 倍 | 速度大幅提升（代价是分割精度略降）|
| **AnomalyNCD 数据加载** | `num_workers` 已 0→4；可再加 prefetch、pin_memory、把 jpg 解码离线成 tensor 缓存 | 训练时 GPU 不等 IO |
| **混合精度训练** | AnomalyNCD 用 AMP（fp16）训练 | 显存减半、速度 ~1.5× |
| **早停 + 增量训练** | 现在固定 100 epoch；用「cls_loss 连续 N 轮不降」早停，或支持从 checkpoint 续训 | 省掉大量无效 epoch |

### 4.2 准度

| 方向 | 具体做法 | 预期收益 |
|------|---------|---------|
| **换更强 backbone** | MuSc 的 CLIP → **DINOv2-large**（权重源 403，需找镜像），对细微缺陷更敏感 | mtd 这类难数据的 anomaly map 区分度直接提升 |
| **换 anomaly map 来源** | 不局限于 MuSc，接 **PatchCore / EfficientAD / RD++**（AnomalyNCD 论文给过这些的 mtd anomaly map 下载）| 用更强的 AD 方法喂 AnomalyNCD |
| **base 集扩充** | 现在 base 固定 AeBAD 4 类；可加入更多已知缺陷类，甚至用户自己标注的少量异常 | 异常先验更丰富，novel 聚类更准 |
| **数据增强** | novel 集做旋转/翻转/颜色抖动，缓解缺陷姿态多样性 | 聚类更鲁棒 |
| **超参调优** | `lr/warmup_teacher_temp_epochs/memax_weight/sup_weight` 这些 SimGCD 系超参对收敛敏感，可做小网格搜索 | 指标提升 |
| **伪标签纠正** | 论文里的 re-corrected pseudo labels 是本方法的重点，可加强 `get_pseudo_label_weights` 里 `anomaly_thred` 的适配 | 减少错误伪标签的误导 |
| **后处理 RsCIN 启用** | MuSc 的 RsCIN（分数传播）默认没在 wrapper 里用，可接入 | anomaly map 更平滑 |

### 4.3 一个值得做的「架构级」改进

当前是 **MuSc 和 AnomalyNCD 硬串联**：MuSc 生成的 anomaly map 质量直接决定 AnomalyNCD 的上限。可以考虑：

1. **多 AD 方法集成**：同时跑 MuSc + PatchCore 等多个 AD 方法，把它们的 anomaly map **融合/投票**，再喂给 AnomalyNCD，降低单一方法对难数据的脆弱性。
2. **可插拔的 anomaly map 源**：把「anomaly map 从哪来」抽象成一个接口，MuSc 只是默认实现之一，用户可以换成自己的 AD 模型或直接上传已有的热力图。
3. **端到端微调**：如果允许少量标注，可以让 MuSc 的 backbone 在目标域上做轻量适配（LoRA/提示微调），从「零样本」变成「少样本」，兼顾成本与准度。

---

## 附：本项目本次调试修复清单（速查）

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | core 流程 cls 不收敛 | base 用了正常类 | `data_bridge.py` base 改为 AeBAD |
| 2 | 标准流程 KeyError | lib 被 core 改坏（base_category/normal_ref）| 恢复原始逻辑 |
| 3 | MEBin 二值化全黑 | anomaly map 逐张 min-max 归一化 | 改全局归一化 |
| 4 | 训练 OOM | batch_size 过大 / MSM 大矩阵 | batch 32→16、MSM 分批 cdist + fp16 |
| 5 | Windows→Linux 崩溃 | 模块名大小写 | 改正确大小写 |
| 6 | 结果写错目录 | `../../../../` 层级算错 | 改 `../../../` |

---

*本文档基于 2026-09 对本项目的完整调试过程整理，算法描述对照 MuSc(CVPR'24)、AnomalyNCD(CVPR'25) 论文及其官方代码。*
