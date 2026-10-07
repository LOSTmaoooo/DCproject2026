#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =================================================================================================
# 实验 0：验证「取景一致性（framing consistency）」假设
#
# 背景：
#   MuSc 的核心假设是「正常 patch 能在其他图里找到很多相似 patch」。
#   这要求同一类的测试图处于**统一取景**下。
#   MVTec 满足（每类图像尺寸完全一致，如 bottle 900x900）。
#   MTD 不满足（583 张测试图 W 105-632 / H 231-403，宽高比 0.37~2.68）。
#
#   而 MuSc 的 CLIP 预处理是 `Resize((518,518))`（tuple → 强制拉伸，不保长宽比），
#   于是 MTD 的图被各向异性拉伸，同一块正常磁瓦表面在不同图里的横向尺度差最多 7~8 倍
#   → 参照系塌陷 → 分数整体抬升到 0.671 并饱和 → 区分度 0.005。
#
# 本脚本做两组对照实验：
#
#   模式 mtd    —— 同一批 MTD 原图，三种 resize 策略对比
#       square      Resize((S,S))     当前行为（各向异性拉伸）
#       letterbox   长边缩到 S + padding（保长宽比，不丢内容）
#       shortcrop   短边缩到 S + center crop（保长宽比，丢边缘内容）
#       → 若 letterbox/shortcrop 明显优于 square，取景假设成立
#
#   模式 mvtec  —— 人为把 MVTec 改造成 MTD 的取景，复现塌陷（判决性实验）
#       ctrl             原图 + square                     （基线，应正常）
#       sim_letterbox    MTD 式裁剪 + letterbox            （裁剪有害吗）
#       sim_square       同一批裁剪 + square                （隔离各向异性）
#       → sim_letterbox 与 sim_square 用**完全相同的裁剪框**，
#         唯一变量是最后一步 resize 的几何 → 干净地隔离出「各向异性」这一个因素。
#
# 用法：
#   python experiments/exp0_framing.py mtd    --classes MT_Blowhole MT_Crack MT_Free ...
#   python experiments/exp0_framing.py mvtec  --category bottle
#   python experiments/exp0_framing.py mvtec  --category bottle --max-per-class 60   # 快速预览
# =================================================================================================

import os
import sys
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------------------------------------
# 路径装配（照搬 core/musc_wrapper.py 的做法，绕开 open_clip 的绝对导入问题）
# -------------------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSC_PATH = os.path.join(PROJECT_ROOT, 'libs', 'MuSc')
BACKBONE_PATH = os.path.join(MUSC_PATH, 'models', 'backbone')
for _p in (BACKBONE_PATH, MUSC_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import models.backbone.open_clip as open_clip
import models.backbone._backbones as _backbones
from models.modules._LNAMD import LNAMD
from models.modules._MSM import MSM

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = os.path.join(PROJECT_ROOT, 'libs', 'AnomalyNCD', 'data')
MTD_ROOT = os.path.join(DATA_ROOT, 'mtd_anomaly_detection')
MVTEC_ROOT = os.path.join(DATA_ROOT, 'mvtec_anomaly_detection')


# =================================================================================================
# 一、取景变换（framing transforms）
# =================================================================================================
def _letterbox(img, S, fill='edge'):
    """长边缩放到 S，其余用 padding 补齐 → 保长宽比，不丢内容。

    fill='edge' 用边缘像素复制填充，避免引入 CLIP 没见过的纯黑边。
    （padding 方式本身值得单独消融，'zero' / 'reflect' 都可以试。）
    """
    w, h = img.size
    scale = S / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = img.resize((nw, nh), Image.BICUBIC)
    a = np.asarray(im)
    pt, pl = (S - nh) // 2, (S - nw) // 2
    pb, pr = S - nh - pt, S - nw - pl
    mode = 'edge' if fill == 'edge' else 'constant'
    a = np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode=mode)
    return Image.fromarray(a)


def _shortcrop(img, S):
    """短边缩放到 S，再 center crop S×S → 保长宽比，无 padding，但会丢弃长边方向的内容。"""
    w, h = img.size
    scale = S / min(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - S) // 2, (nh - S) // 2
    return im.crop((left, top, left + S, top + S))


def _square(img, S):
    """当前 MuSc 的行为：精确缩放到 S×S，不保长宽比。"""
    return img.resize((S, S), Image.BICUBIC)


def _to_tensor(img):
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    a = (a - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(a).permute(2, 0, 1)


FRAMINGS = {
    'square': _square,
    'letterbox': _letterbox,
    'shortcrop': _shortcrop,
}


# =================================================================================================
# 二、数据索引
# =================================================================================================
def build_mtd_index(classes=None):
    """MTD: test/<class>/*.jpg + ground_truth/<class>/*.png"""
    out = []
    test_root = os.path.join(MTD_ROOT, 'test')
    gt_root = os.path.join(MTD_ROOT, 'ground_truth')
    for cls in sorted(os.listdir(test_root)):
        if classes and cls not in classes:
            continue
        d = os.path.join(test_root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.jpg', '.png')):
                continue
            gt = os.path.join(gt_root, cls, os.path.splitext(f)[0] + '.png')
            out.append({
                'path': os.path.join(d, f),
                'gt': gt if os.path.exists(gt) else None,
                'cls': cls,
                'is_anomaly': int(cls != 'MT_Free'),
            })
    return out


def build_mvtec_index(category):
    """MVTec: <cat>/test/<defect_type>/*.png + ground_truth/<defect_type>/*_mask.png"""
    out = []
    test_root = os.path.join(MVTEC_ROOT, category, 'test')
    gt_root = os.path.join(MVTEC_ROOT, category, 'ground_truth')
    for dtype in sorted(os.listdir(test_root)):
        d = os.path.join(test_root, dtype)
        if not os.path.isdir(d):
            continue
        is_anom = int(dtype != 'good')
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.png', '.jpg')):
                continue
            gt = None
            if is_anom:
                cand = os.path.join(gt_root, dtype, os.path.splitext(f)[0] + '_mask.png')
                if os.path.exists(cand):
                    gt = cand
            out.append({
                'path': os.path.join(d, f),
                'gt': gt,
                'cls': dtype,
                'is_anomaly': is_anom,
            })
    return out


def defect_stats(index):
    """统计每个类别的缺陷面积占比（缺陷像素数 / 整图像素数）。"""
    per_cls = defaultdict(list)
    for it in index:
        if not it['is_anomaly'] or not it['gt']:
            continue
        m = np.array(Image.open(it['gt']).convert('L')) > 127
        if m.size == 0:
            continue
        per_cls[it['cls']].append(m.sum() / m.size)
    return per_cls


# =================================================================================================
# 三、MTD 式裁剪（用于 mvtec 模式的 sim_* 两组，保证两组裁同一批框）
# =================================================================================================
def mtd_aspect_pool():
    """从真实 MTD 采样宽高比分布。MTD 不在本地时退回硬编码的实测分布。"""
    ars = []
    test_root = os.path.join(MTD_ROOT, 'test')
    if os.path.isdir(test_root):
        for cls in os.listdir(test_root):
            d = os.path.join(test_root, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                try:
                    w, h = Image.open(os.path.join(d, f)).size
                    ars.append(w / h)
                except Exception:
                    pass
    if not ars:  # 实测：min 0.37 median 0.84 max 2.68 std 0.61
        ars = [0.37, 0.42, 0.60, 0.84, 0.84, 1.19, 1.60, 2.38, 2.68]
    return np.array(ars, dtype=np.float32)


def _defect_bbox(it):
    """返回缺陷的包围盒 (x0,y0,x1,y1) 与质心；正常图返回 None。"""
    if not it['is_anomaly'] or not it['gt']:
        return None
    m = np.array(Image.open(it['gt']).convert('L')) > 127
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def make_crop_plan(items, aspect_pool, rng, area_frac_range=(0.25, 0.75)):
    """为每张图生成裁剪框 (x0,y0,x1,y1)，模拟 MTD「大视野任意区域」的取景。

    要点（修正版）：
      - **裁剪只会让缺陷占比变大**，所以不能靠裁剪切去「制造」MTD 那种 0.1% 的小缺陷占比。
        MTD 的缺陷占比小，恰恰因为它是**大视野**而非紧裁剪。因此这里只控制：
        (a) 宽高比 ar ~ MTD 实测分布   ← 各向异性的来源
        (b) 裁剪面积占原图比例        ← 保证不把产品裁没
      - 异常图：裁剪框**必须完整包含缺陷包围盒**，以缺陷包围盒中心为中心，再平移贴边。
      - 正常图：随机中心。
    """
    plan = []
    for it in items:
        img = Image.open(it['path'])
        W, H = img.size
        ar = float(rng.choice(aspect_pool))
        af = rng.uniform(*area_frac_range)

        cw = math.sqrt(af * W * H * ar)
        ch = cw / ar
        cw, ch = min(cw, W), min(ch, H)

        bb = _defect_bbox(it)
        if bb is not None:
            bx0, by0, bx1, by1 = bb
            bw, bh = bx1 - bx0, by1 - by0
            # 保证裁剪框两个方向都装得下整个缺陷包围盒
            cw = max(cw, bw, bh * ar)
            ch = cw / ar
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        else:
            cx, cy = rng.uniform(0.35 * W, 0.65 * W), rng.uniform(0.35 * H, 0.65 * H)

        # 收缩到画布内
        s = min(1.0, W / cw, H / ch)
        cw, ch = cw * s, ch * s
        # 极端宽高比 + 大缺陷时，几何上无法同时满足 → 优先保缺陷完整，接受宽高比偏离
        if bb is not None and (cw < bw - 1e-6 or ch < bh - 1e-6):
            cw = min(float(W), max(bw, cw))
            ch = min(float(H), max(bh, ch))

        x0 = min(max(cx - cw / 2, 0), W - cw)
        y0 = min(max(cy - ch / 2, 0), H - ch)
        plan.append(('crop', (int(x0), int(y0), int(x0 + cw), int(y0 + ch))))
    return plan


def make_pad_plan(items, aspect_pool, rng):
    """为每张图生成 padding 方案：把图 pad 成目标宽高比，再走正方形 resize。

    ★ 这是「零信息损失」的几何对照 ★
    不做任何裁剪，只是把画布补成 MTD 式的非正方形宽高比，
    随后 `Resize((S,S))` 就会产生各向异性拉伸。
    因此它**单独隔离出「各向异性几何」这一个变量**，内容一个像素都没丢。
    """
    plan = []
    for it in items:
        img = Image.open(it['path'])
        W, H = img.size
        ar = float(rng.choice(aspect_pool))
        if W / H < ar:          # 太窄 → 左右补
            nw, nh = int(round(H * ar)), H
        else:                   # 太宽 → 上下补
            nw, nh = W, int(round(W / ar))
        plan.append(('pad', (nw, nh)))
    return plan


def apply_plan(img, spec):
    """按 plan 里的一条记录处理 PIL 图像。"""
    if spec is None:
        return img
    mode, p = spec
    if mode == 'crop':
        return img.crop(p)
    if mode == 'pad':
        nw, nh = p
        a = np.asarray(img.convert('RGB'))
        pt, pl = (nh - a.shape[0]) // 2, (nw - a.shape[1]) // 2
        pb, pr = nh - a.shape[0] - pt, nw - a.shape[1] - pl
        a = np.pad(a, ((pt, pb), (pl, pr), (0, 0)), mode='edge')
        return Image.fromarray(a)
    return img


# =================================================================================================
# 四、MuSc 前向：提特征 → LNAMD → MSM
# =================================================================================================
class Backbone:
    def __init__(self, model_name, image_size, device):
        self.model_name = model_name
        self.image_size = image_size
        self.device = device
        if 'dino' in model_name:
            self.model = _backbones.load(model_name).to(device).eval()
            self.preprocess = None
        else:
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, image_size, pretrained='openai')
            self.model = self.model.to(device).eval()
            self.preprocess = None
        self.features_list = [l + 1 for l in [5, 11, 17, 23]]

    @torch.no_grad()
    def embed(self, batch):
        if 'dinov2' in self.model_name:
            pt = self.model.get_intermediate_layers(
                x=batch, n=[l - 1 for l in self.features_list], return_class_token=False)
            pt = [p.cpu() for p in pt]
            fake = [torch.zeros_like(p)[:, 0:1, :] for p in pt]
            pt = [torch.cat([fake[i], pt[i]], dim=1) for i in range(len(pt))]
        elif 'dino' in self.model_name:
            pt_all = self.model.get_intermediate_layers(x=batch, n=max(self.features_list))
            pt = [pt_all[l - 1].cpu() for l in self.features_list]
        else:
            _, pt = self.model.encode_image(batch, self.features_list)
            pt = [pt[l].cpu() for l in range(len(self.features_list))]
        return pt


def run_musc(backbone, items, framing, crop_plan, S, r_list, device, chunk_bs=4):
    """对一批图跑 MuSc，返回 (N, S, S) 的异常图和每张图的 feature 张量缓存。"""
    # --- 提特征 ---
    feats = []
    batch, n_batch = [], 0
    for i, it in enumerate(tqdm(items, desc=f'  extract[{framing}]', leave=False)):
        img = apply_plan(Image.open(it['path']).convert('RGB'),
                         None if crop_plan is None else crop_plan[i])
        batch.append(_to_tensor(FRAMINGS[framing](img, S)))
        n_batch += 1
        if n_batch == chunk_bs or i == len(items) - 1:
            x = torch.stack(batch).to(device)
            with torch.no_grad():
                feats.append(backbone.embed(x))
            batch, n_batch = [], 0

    # --- LNAMD + MSM，按 r 求平均 ---
    feature_dim = feats[0][0].shape[-1]
    maps_r = []
    for r in r_list:
        ln = LNAMD(device=device, r=r, feature_dim=feature_dim,
                   feature_layer=backbone.features_list)
        Z_layers = defaultdict(list)
        for bf in feats:
            bf = [f.to(device) for f in bf]
            with torch.no_grad():
                f = ln._embed(bf)
                f = f / f.norm(dim=-1, keepdim=True)
                for l in range(len(backbone.features_list)):
                    Z_layers[l].append(f[:, :, l, :])
        maps_l = []
        for l in Z_layers:
            Z = torch.cat(Z_layers[l], dim=0).to(device, dtype=torch.float32)
            m = MSM(Z=Z, device=device, topmin_min=0, topmin_max=0.3)
            maps_l.append(m.unsqueeze(0).cpu())
            del Z
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        maps_r.append(torch.mean(torch.cat(maps_l, 0), 0))
    m = torch.mean(torch.stack(maps_r, 0), 0)          # (N, L_patches)

    N, L = m.shape
    Hh = int(math.sqrt(L))
    m = F.interpolate(m.view(N, 1, Hh, Hh), size=S, mode='bilinear', align_corners=True)
    return m.squeeze(1).numpy()


# =================================================================================================
# 五、评测
# =================================================================================================
def _ks(a, b):
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, 'right') / len(a)
                               - np.searchsorted(b, grid, 'right') / len(b))))


def evaluate(maps, items, tag, frame='max'):
    """maps: (N,S,S)。

    打印三层：
      ALL                 —— 全体正常 vs 全体异常
      <类名> vs 正常       —— 每个异常类单独和正常类比（★ MTD 上最有信息量的一行）
      <类名> 均分          —— 该类自身均分（MTD 每类天然单标签，AUROC 无定义，看均分）
    """
    n = len(items)
    scores = maps.reshape(n, -1).max(-1) if frame == 'max' else maps.reshape(n, -1).mean(-1)
    labels = np.array([it['is_anomaly'] for it in items])
    classes = np.array([it['cls'] for it in items])
    has_norm = (labels == 0).any()
    g_mean_n = scores[labels == 0].mean() if has_norm else float('nan')

    print(f'\n[{tag}]  图像级分数 = map 的 {frame}')
    if has_norm and (labels == 1).any():
        a, b = scores[labels == 1], scores[labels == 0]
        print(f'  {"ALL vs 正常":22s} n={n:4d}  正常{b.mean():.3f} 异常{a.mean():.3f} '
              f'区分度{a.mean() - b.mean():+.3f} AUROC {roc_auc_score(labels, scores):.4f} '
              f'KS {_ks(a, b):.3f}')

    for c in sorted(set(classes)):
        mc = classes == c
        nc, na = int((mc & (labels == 0)).sum()), int((mc & (labels == 1)).sum())
        s = scores[mc]
        # 该异常类 vs 全部正常图 —— MTD 上这就是「这一类测不测得出来」
        if has_norm and na > 0 and nc == 0:
            sub = mc | (labels == 0)
            y, sc = labels[sub], scores[sub]
            a, b = sc[y == 1], sc[y == 0]
            print(f'  {c + " vs 正常":22s} n={na:4d}  正常{b.mean():.3f} 异常{a.mean():.3f} '
                  f'区分度{a.mean() - b.mean():+.3f} AUROC {roc_auc_score(y, sc):.4f} '
                  f'KS {_ks(a, b):.3f}  ←')
        else:
            rel = '' if not has_norm or nc > 0 else f'  相对正常 {s.mean() - g_mean_n:+.3f}'
            print(f'  {c:22s} n={int(mc.sum()):4d}  均分 {s.mean():.3f}{rel}'
                  f'   [单一标签，AUROC 无定义]')

    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float('nan')


# =================================================================================================
# 六、主流程
# =================================================================================================
def cmd_mtd(a):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    index = build_mtd_index(a.classes)
    ds = defect_stats(index)
    print('\nMTD 缺陷面积占比（中位数）：')
    for c, v in sorted(ds.items()):
        print(f'  {c:14s} median {np.median(v):.4f}  → @512 约 '
              f'{int(math.sqrt(np.median(v) * 512 * 512))}x'
              f'{int(math.sqrt(np.median(v) * 512 * 512))}')
    if a.max_per_class:
        index = _subsample(index, a.max_per_class, 42)
    print(f'\n共 {len(index)} 张')

    bk = Backbone(a.backbone, a.image_size, device)
    res = {}
    for framing in a.framings:
        m = run_musc(bk, index, framing, None, a.image_size, a.r_list, device)
        res[framing] = evaluate(m, index, f'MTD / {framing}')
        np.save(os.path.join(a.out, f'mtd_{framing}.npy'), m)
    print('\n=== MTD 结论 ===')
    for k, v in res.items():
        print(f'  {k:12s} AUROC {v:.4f}')


def cmd_mvtec(a):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    index = build_mvtec_index(a.category)
    if a.max_per_class:
        index = _subsample(index, a.max_per_class, 42)
    print(f'\n{len(index)} 张，类别：', sorted(set(it["cls"] for it in index)))

    bk = Backbone(a.backbone, a.image_size, device)
    pool = mtd_aspect_pool()
    rng = random.Random(a.seed)
    crop_plan = make_crop_plan(index, pool, rng)
    pad_plan = make_pad_plan(index, pool, random.Random(a.seed))

    arms = [
        # name,            plan,        framing,     说明
        ('ctrl',           None,        'square'),    # 基线
        ('pad_warp',       pad_plan,    'square'),    # ★ 纯几何：只 pad 成 MTD 宽高比，零裁剪零信息损失
        ('sim_letterbox',  crop_plan,   'letterbox'), # MTD 式裁剪 + 保长宽比（对照：裁剪本身有害吗）
        ('sim_square',     crop_plan,   'square'),    # 同一批裁剪 + 各向异性拉伸（真实 MTD 模拟）
    ]
    res = {}
    for name, plan, framing in arms:
        m = run_musc(bk, index, framing, plan, a.image_size, a.r_list, device)
        res[name] = evaluate(m, index, f'{a.category} / {name}')
        np.save(os.path.join(a.out, f'{a.category}_{name}.npy'), m)

    print('\n=== MVTec 判决性结论 ===')
    print(f'  ctrl           (原图 + 正方形，基线)          AUROC {res["ctrl"]:.4f}')
    print(f'  pad_warp       (pad 成 MTD 宽高比 + 正方形)   AUROC {res["pad_warp"]:.4f}'
          f'    Δ={res["pad_warp"] - res["ctrl"]:+.4f}')
    print(f'  sim_letterbox  (MTD 式裁剪 + 保长宽比)        AUROC {res["sim_letterbox"]:.4f}'
          f'    Δ={res["sim_letterbox"] - res["ctrl"]:+.4f}')
    print(f'  sim_square     (同一批裁剪 + 各向异性拉伸)    AUROC {res["sim_square"]:.4f}'
          f'    Δ={res["sim_square"] - res["ctrl"]:+.4f}')
    print()
    print('  读法：')
    print('    pad_warp 掉点明显          → **纯几何（各向异性）足以让 MuSc 失效**，与裁剪/内容无关')
    print('    sim_letterbox ≈ ctrl，')
    print('    而 sim_square 掉点明显     → 裁剪无害，元凶是最后一步 resize 的几何')
    print('    两者都掉点                 → 取景的两部分（内容异质 + 几何失真）都在起作用')


def _subsample(index, k, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in index:
        by[it['cls']].append(it)
    out = []
    for c, v in by.items():
        out.extend(v if len(v) <= k else rng.sample(v, k))
    return out


def main():
    ap = argparse.ArgumentParser(description='实验0：验证取景一致性假设')
    sub = ap.add_subparsers(dest='mode', required=True)

    def common(p):
        p.add_argument('--backbone', default='ViT-L-14-336')
        p.add_argument('--image_size', type=int, default=518)
        p.add_argument('--r_list', type=int, nargs='+', default=[1, 3, 5])
        p.add_argument('--max-per-class', type=int, default=0,
                       help='每类最多取多少张（0=不限，预览时用 40~60）')
        p.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'experiments', 'out'))

    m = sub.add_parser('mtd')
    common(m)
    m.add_argument('--classes', nargs='+', default=None)
    m.add_argument('--framings', nargs='+', default=['square', 'letterbox', 'shortcrop'])
    m.set_defaults(func=cmd_mtd)

    v = sub.add_parser('mvtec')
    common(v)
    v.add_argument('--category', default='bottle')
    v.add_argument('--seed', type=int, default=42)
    v.set_defaults(func=cmd_mvtec)

    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.func(a)


if __name__ == '__main__':
    main()
