# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =================================================================================================
# 模块：MuSc 模型封装器 (MuSc Wrapper)
# 功能：封装 MuSc 算法的图像预处理、特征提取和异常图生成逻辑，使其易于被外部引擎调用。
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 路径配置与动态导入
# -------------------------------------------------------------------------------------------------
# 动态将 libs/MuSc 添加到系统路径，以便可以导入原始 MuSc 库中的模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSC_PATH = os.path.join(PROJECT_ROOT, 'libs', 'MuSc')
sys.path.append(MUSC_PATH)

# 从 MuSc 库中导入核心模型和工具
try:
    from models.musc import MuSc
    from utils.load_config import load_yaml
    # 直接导入 LNAMD (局部邻域聚合) 和 MSM (多尺度映射) 模块，这在后续手动计算流程中需要使用
    from models.modules._LNAMD import LNAMD
    from models.modules._MSM import MSM
except ImportError as e:
    print(f"[MuScWrapper] Import Error: {e}. Please check permissions or path: {MUSC_PATH}")

# ==========================================
# 1. 自定义推理数据集 (Inference Dataset)
# ==========================================
class InferenceDataset(Dataset):
    """
    用于推理的简单数据集类。
    
    目的：
    原始的 MVTecDataset 通常需要 Ground Truth (掩模文件) 才能加载，而我们在实际推理/应用阶段
    只有输入图片，没有标签。因此创建此类来加载图片并提供类似的接口。
    """
    def __init__(self, image_dir, resize=256, imagesize=224, clip_transformer=None):
        self.image_dir = image_dir
        # 支持的图片扩展名
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]
        
        # 图像预处理流水线
        # 如果模型提供了特定的 transformer (如 CLIP)，则使用它；否则使用标准的 ImageNet 归一化处理
        if clip_transformer is None:
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = clip_transformer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        返回单个数据样本。
        返回字典包含： 'image' (张量), 'mask' (全0占位符), 'image_path'
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        # 创建一个假的掩模（全0），仅为了保持接口一致性，实际上推理不使用
        dummy_mask = torch.zeros([1, image_tensor.shape[1], image_tensor.shape[2]])
        
        return {
            "image": image_tensor,
            "mask": dummy_mask,
            "is_anomaly": 0, # 默认为0，未知
            "image_path": img_path
        }

# ==========================================
# 2. MuSc 封装类 (MuSc Wrapper)
# ==========================================
class MuScWrapper:
    """
    MuSc 算法的主要接口类。
    
    主要职责：
    1. 初始化模型配置和计算设备 (CPU/GPU)。
    2. 加载预训练的 Backbone (DINOv2 或 CLIP)。
    3. 提供 generate_anomaly_maps 接口，执行批量特征提取和评分计算。
    """
    
    def __init__(self, config_path, device=None):
        print(f"[MuScWrapper] Loading configuration from: {config_path}")
        self.cfg = load_yaml(config_path)
        
        # -------------------
        # 设备 (Device) 设置
        # -------------------
        if device is None:
             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
             self.device = torch.device(f"cuda:{device}")
        else:
             self.device = torch.device(device)
             
        # 这里的 hack 是为了适配原始代码可能的 bug 或严格类型检查
        # MuSc 原始代码可能期望 'device' 配置项是整数卡号，而我们可能使用 cpu
        if self.device.type == 'cpu':
             self.cfg['device'] = 'cpu' 
        else:
             self.cfg['device'] = self.device.index if self.device.index is not None else 0

        # -------------------
        # 模型初始化
        # -------------------
        print(f"[MuScWrapper] Initializing MuSc Model on {self.device}...")
        self.musc_model = MuSc(self.cfg, seed=42)
        
        # 提取底层的 Backbone 模型 (DINO 或 CLIP)，用于单独的数据流控制
        if hasattr(self.musc_model, 'dino_model'):
            self.backbone = self.musc_model.dino_model
            self.model_type = 'dino'
        elif hasattr(self.musc_model, 'clip_model'):
            self.backbone = self.musc_model.clip_model
            self.model_type = 'clip'
        else:
            raise ValueError("Unknown backbone type in MuSc model")

        # 获取预处理函数和特征层配置
        self.preprocess = self.musc_model.preprocess
        self.features_list = self.musc_model.features_list # 需要提取特征的层索引列表
        self.r_list = self.musc_model.r_list               # 聚合度参数列表 (aggregation degrees)
        self.batch_size = self.cfg['models']['batch_size']
        self.image_size = self.cfg['datasets']['img_resize']
        
    def _extract_features(self, images):
        """
        内部辅助方法：使用 Backbone 提取中间层特征。
        根据不同的模型类型 (DINOv2, DINO, CLIP) 调用不同的提取逻辑。
        """
        # 情况 1: DINOv2
        if 'dinov2' in self.musc_model.model_name and self.model_type == 'dino':
            # 获取指定层的中间输出
            patch_tokens = self.backbone.get_intermediate_layers(x=images, n=[l-1 for l in self.features_list], return_class_token=False)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            # 手动添加 pseudo-CLS token 以保持形状一致性 (如果需要)
            fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
            patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
            
        # 情况 2: 普通 DINO
        elif 'dino' in self.musc_model.model_name and self.model_type == 'dino':
            patch_tokens_all = self.backbone.get_intermediate_layers(x=images, n=max(self.features_list))
            patch_tokens = [patch_tokens_all[l-1].cpu() for l in self.features_list]
            
        # 情况 3: CLIP
        else: 
            _, patch_tokens = self.backbone.encode_image(images, self.features_list)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            
        return patch_tokens

    def generate_anomaly_maps(self, input_dir, output_dir):
        """
        核心方法：对目录下的所有图像生成异常热力图 (Anomaly Maps)。
        
        过程：
        1. 构建 DataLoader 加载图片。
        2. 批量提取特征 (Batch Feature Extraction)。
        3. 还原到 list of features 方便后续处理。
        4. 使用 LNAMD 模块在不同聚合度 r 下计算特征的邻域异常。
        5. 使用 MSM 模块对结果进行多尺度融合评分。
        6. 上采样并保存结果为 .npy 文件。
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[MuScWrapper] Starting batch generation for: {input_dir}")
        
        # 1. 数据准备
        dataset = InferenceDataset(
            image_dir=input_dir, 
            resize=self.image_size, 
            imagesize=self.image_size,
            clip_transformer=self.preprocess
        )
        
        if len(dataset) == 0:
            print(f"[MuScWrapper] Warning: No images found in {input_dir}")
            return []
            
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # 2. 阶段一：特征提取 (Feature Extraction)
        print("[MuScWrapper] Step 1/2: Extracting features...")
        patch_tokens_list = [] # 存储每批次的特征
        image_paths_all = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                images = batch["image"].to(self.device)
                paths = batch["image_path"]
                
                # 提取得到 [Layer1_Feature(B,L,C), Layer2_Feature(B,L,C)...]
                batch_features = self._extract_features(images)
                
                patch_tokens_list.append(batch_features) 
                image_paths_all.extend(paths)
                
        # 3. 阶段二：计算异常分数 (LNAMD + MSM)
        print("[MuScWrapper] Step 2/2: Computing Scores (LNAMD + MSM)...")
        
        # anomaly_maps_r 用于存储不同 r (aggregation degree) 下的计算结果
        anomaly_maps_r = torch.tensor([]).double() 
        
        # 获取特征维度 (C)，用于初始化 LNAMD
        feature_dim = patch_tokens_list[0][0].shape[-1]
        
        # 遍历配置中定义的每个聚合度 r
        for r in self.r_list:
            print(f'Processing aggregation degree r={r}...')
            # 初始化 LNAMD 模块
            LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)
            
            Z_layers = {} # 按层存储嵌入后的特征
            
            # 使用 LNAMD 处理提取出的特征
            for batch_idx, batch_feats in enumerate(patch_tokens_list):
                # 将特征移动到 GPU
                batch_feats_gpu = [f.to(self.device) for f in batch_feats]
                
                with torch.no_grad():
                     # LNAMD._embed 进行邻域聚合嵌入
                     # 输出形状: (B, N_patches, num_layers, C_out)
                    features = LNAMD_r._embed(batch_feats_gpu)
                    # 归一化
                    features /= features.norm(dim=-1, keepdim=True)
                    
                    # 按层拆分并存储
                    for l_idx in range(len(self.features_list)):
                        l_key = str(l_idx)
                        if l_key not in Z_layers: Z_layers[l_key] = []
                        Z_layers[l_key].append(features[:, :, l_idx, :])
            
            # MSM (Multi-Scale Map) 评分计算
            anomaly_maps_l = torch.tensor([]).double()
            
            for l_key in Z_layers.keys():
                # 拼接所有批次 -> (Total_Images, N_patches, C)
                Z = torch.cat(Z_layers[l_key], dim=0).to(self.device)
                
                # 计算余弦相似度等指标，得到分数 (Total_Images, N_patches)
                scores = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                
                # 收集每一层的结果
                anomaly_maps_l = torch.cat((anomaly_maps_l, scores.unsqueeze(0).cpu()), dim=0)
                del Z
                torch.cuda.empty_cache()
            
            # 对多个层的结果取平均
            anomaly_maps_l = torch.mean(anomaly_maps_l, dim=0) 
            # 收集当前 r 的结果
            anomaly_maps_r = torch.cat((anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0)

        # 4. 融合所有 r 的结果
        # 对不同 r 维度的结果取平均，作为最终的 anomaly map
        anomaly_maps_final = torch.mean(anomaly_maps_r, dim=0).to(self.device)
        
        # 5. 上采样与保存
        N, L_patches = anomaly_maps_final.shape
        H_feat = int(np.sqrt(L_patches)) # 计算特征图的宽高 (例如 14x14)
        
        # 双线性插值上采样到原图大小
        anomaly_maps_resized = F.interpolate(
            anomaly_maps_final.view(N, 1, H_feat, H_feat),
            size=self.image_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        # 保存结果
        print(f"[MuScWrapper] Saving maps to {output_dir}")
        saved_paths = []
        maps_np = anomaly_maps_resized.cpu().numpy() # 转为 numpy 数组 (N, 1, H, W)
        
        for i, img_path in enumerate(image_paths_all):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_map.npy")
            
            # 保存单张热力图
            map_data = maps_np[i, 0, :, :]
            np.save(save_path, map_data)
            saved_paths.append(save_path)
            
        print(f"[MuScWrapper] Generated {len(saved_paths)} maps.")
        return saved_paths


