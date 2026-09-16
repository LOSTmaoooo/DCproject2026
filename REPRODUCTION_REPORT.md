# 复现报告：MuSc + AnomalyNCD 在 MTD 上的 Novel Anomaly Class Discovery

> 日期：2026-09-13
> 结论：**成功复现论文官方结果，NMI 略超、ARI/F1 持平 SOTA。**
> 状态：可作为稳定基线留存，后续优化以此为对照。

---

## 1. 复现目标

复现论文 *AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios*（CVPR 2025）中，
**MuSc（zero-shot 异常检测）+ AnomalyNCD（新类发现）** 在 **MTD（磁瓦缺陷）** 数据集上的无监督多类分类结果。

- base（有标签先验）：AeBAD-S 4 类 —— `ablation / breakdown / fracture / groove`
- novel（无标签，待发现）：MTD 5 类 —— `MT_Blowhole / MT_Break / MT_Crack / MT_Fray / MT_Uneven`
- 异常图来源：MuSc（zero-shot，无训练）

## 2. 实验配置

| 项 | 值 |
|---|---|
| backbone | `dino_vitb8`（DINOv2 ViT-B/8） |
| epochs | 100 |
| batch_size | 16（本次从 32 调低，缓解显存） |
| num_workers | 4 |
| lr / momentum / weight_decay | 0.003 / 0.9 / 5e-5 |
| seed | 3407 |
| loss 权重 | sup_weight 0.3，memax_weight 4，anomaly_thred 0.5 |
| 教师温度 | teacher_temp 0.04，warmup 40 epoch |
| binarization | sample_rate 4，min_interval_len 4，erode True |

novel 类样本数（原始图）：MT_Blowhole 115 / MT_Break 85 / MT_Crack 57 / MT_Fray 32 / MT_Uneven 103（共 392 张，裁剪后成子图参与训练）。

## 3. 复现结果 vs 论文官方

论文官方（README，MuSc+AnomalyNCD @ MTD）：

| 指标 | NMI | ARI | F1 |
|---|---|---|---|
| 论文官方 | 0.268 | 0.228 | 0.509 |

本次复现：

| 阶段 | NMI | ARI | F1 |
|---|---|---|---|
| sub-image（最终 epoch 99） | 0.245 | 0.183 | 0.465 |
| **region-merge（最终，image-level）** | **0.295** | **0.223** | **0.500** |
| sub-image（epoch 39 峰值） | 0.287 | 0.219 | 0.550 |

**对比结论**：region-merge 的 NMI **0.295 略超官方 0.268**；ARI 0.223 与 0.228 基本持平；F1 0.500 与 0.509 基本持平。
即：**完整复现论文 SOTA 水平，无显著偏差。**

> 说明：MTD 是难数据集，论文表中该数据集所有方法指标都偏低（最强半监督 PatchCore+AnomalyNCD 也仅 NMI 0.38）。
> 0.24–0.30 的 NMI 是这套方法在 MTD 上的合理上限，并非实现错误。

## 4. 训练过程与收敛

| loss 分量 | 起点（ep0） | 终点（ep99） | 说明 |
|---|---|---|---|
| total | 4.35 | ~3.2 | 平稳下降 |
| cls_loss | 2.17 | ~0.002 | base 四分类基本学满 |
| cluster_loss | 2.35 | ~1.3 | 缓慢下降 |
| contrastive_loss | 3.15 | ~2.5 | 收敛平台，属正常 |

**关键现象**：聚类指标在 **epoch 39 见顶（NMI 0.287 / F1 0.550）后开始回落**，最终收敛到 NMI 0.245。
该转折点与配置中 `warmup_teacher_temp_epochs: 40` 高度重合，是后续优化的重点线索（见第 5 节）。

## 5. 待优化方向（不修改 lib 代码）

1. **换上游 AD 出 anomaly map**（最高收益、最低风险）：论文半监督表显示 PatchCore / CPR 出 map 时 MTD NMI 可达 0.38–0.39，明显高于 MuSc 的 0.268。
2. **解决 epoch-39 后性能退化**：研究教师温度 warmup 与过拟合的关系，可能一个调度改动即可稳定住峰值。
3. **类不平衡**：MT_Fray 仅 32 张，小类在 NCD 中天然吃亏，可改进 pseudo-label / 采样策略。

## 6. 运行信息

- 完整流水线日志：`/root/autodl-tmp/DCproject/train_log.txt`
- 训练耗时：100 epoch 约 2.5 小时（~88s/epoch，110 iters × batch 16）
- 硬件：单卡 CUDA GPU（cuda:0）
- 复现所用代码提交：`bdcf3ce`（fix）+ `180e3c2`（refactor）
