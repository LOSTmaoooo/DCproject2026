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

# Dynamically add libs path to ensure MuSc modules can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSC_PATH = os.path.join(PROJECT_ROOT, 'libs', 'MuSc')
sys.path.append(MUSC_PATH)

# Import MuSc core model and config loader
from models.musc import MuSc
from utils.load_config import load_yaml
# Import necessary modules directly since they are not methods of MuSc class
from models.modules._LNAMD import LNAMD
from models.modules._MSM import MSM

# ==========================================
# 1. Custom Inference Dataset
# ==========================================
# Purpose: Bypass the original MVTecDataset's strict requirement for Ground Truth (masks).
class InferenceDataset(Dataset):
    def __init__(self, image_dir, resize=256, imagesize=224, clip_transformer=None):
        self.image_dir = image_dir
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(valid_exts)
        ]
        
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
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        dummy_mask = torch.zeros([1, image_tensor.shape[1], image_tensor.shape[2]])
        
        return {
            "image": image_tensor,
            "mask": dummy_mask,
            "is_anomaly": 0,
            "image_path": img_path
        }

# ==========================================
# 2. MuSc Wrapper
# ==========================================
class MuScWrapper:
    def __init__(self, config_path, device=None):
        print(f"[MuScWrapper] Loading configuration from: {config_path}")
        self.cfg = load_yaml(config_path)
        
        # Determine device
        if device is None:
             self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
             self.device = torch.device(f"cuda:{device}")
        else:
             self.device = torch.device(device)
             
        # Override config device to match what we perform
        # MuSc uses int or str in its __init__, we will handle it by patching the cfg if needed
        # But MuSc.__init__ logic: self.device = torch.device("cuda:{}".format(cfg['device']) ...
        # So we better set cfg['device'] to a simpler value or let MuSc init safely
        if self.device.type == 'cpu':
             self.cfg['device'] = 'cpu' # Invalid for MuSc probably as it expects int index usually
        else:
             self.cfg['device'] = self.device.index if self.device.index is not None else 0

        # Instantiate original MuSc model
        print(f"[MuScWrapper] Initializing MuSc Model on {self.device}...")
        self.musc_model = MuSc(self.cfg, seed=42)
        
        # Extract model components for direct use
        if hasattr(self.musc_model, 'dino_model'):
            self.backbone = self.musc_model.dino_model
            self.model_type = 'dino'
        elif hasattr(self.musc_model, 'clip_model'):
            self.backbone = self.musc_model.clip_model
            self.model_type = 'clip'
        else:
            raise ValueError("Unknown backbone type in MuSc model")

        self.preprocess = self.musc_model.preprocess
        self.features_list = self.musc_model.features_list
        self.r_list = self.musc_model.r_list
        self.batch_size = self.cfg['models']['batch_size']
        self.image_size = self.cfg['datasets']['img_resize']
        
    def _extract_features(self, images):
        """Helper to extract features using the backbone"""
        if 'dinov2' in self.musc_model.model_name and self.model_type == 'dino':
             # Logic from MuSc.py line ~165
            patch_tokens = self.backbone.get_intermediate_layers(x=images, n=[l-1 for l in self.features_list], return_class_token=False)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens]
            patch_tokens = [torch.cat([fake_cls[i], patch_tokens[i]], dim=1) for i in range(len(patch_tokens))]
            
        elif 'dino' in self.musc_model.model_name and self.model_type == 'dino':
             # Logic from MuSc.py line ~171
            patch_tokens_all = self.backbone.get_intermediate_layers(x=images, n=max(self.features_list))
            patch_tokens = [patch_tokens_all[l-1].cpu() for l in self.features_list]
            
        else: # clip
             # Logic from MuSc.py line ~175
            _, patch_tokens = self.backbone.encode_image(images, self.features_list)
            patch_tokens = [patch_tokens[l].cpu() for l in range(len(self.features_list))]
            
        return patch_tokens

    def generate_anomaly_maps(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[MuScWrapper] Starting batch generation for: {input_dir}")
        
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
        
        # 1. Feature Extraction Loop
        print("[MuScWrapper] Step 1/2: Extracting features...")
        patch_tokens_list = [] # List of lists of tensors
        image_paths_all = []
        
        # self.backbone.eval() # Backbone is usually already in eval mode
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting"):
                images = batch["image"].to(self.device)
                paths = batch["image_path"]
                
                # Extract features for this batch
                # Returns a list of tensors [Layer1_feat(B, L, C), Layer2_feat...]
                batch_features = self._extract_features(images)
                
                # For simplicity, if multiple layers, we might zip them. 
                # MuSc logic appends the list-of-layers-features for each image.
                # Here batch_features is list of (B, L, C). MuSc stores list of list per image.
                # Let's verify MuSc structure: patch_tokens_list.append(patch_tokens) where patch_tokens is list of (1, L, C)??
                # No, in MuSc: patch_tokens_list.append(patch_tokens) inside the loop over dataloader.
                # So patch_tokens_list is [Batch1_feats, Batch2_feats...]
                patch_tokens_list.append(batch_features) 
                image_paths_all.extend(paths)
                
        # 2. LNAMD & MSM Logic (Reimplemented from MuSc.py)
        # MuSc iterates over r_list (aggregation degrees)
        # Flatten the list structure: We need a list where each element corresponds to one image, 
        # and contains a list of features for its layers.
        # Currently patch_tokens_list is [ [Layer1(B), Layer2(B)], [Layer1(B), ...] ]
        # We need to reshape this to be friendly for LNAMD.
        
        # Let's reconstruct the full list of features per image to match MuSc Logic exactly
        # MuSc: patch_tokens_list is a list of batches. output of dataloader loop.
        
        print("[MuScWrapper] Step 2/2: Computing Scores (LNAMD + MSM)...")
        
        # Initialize result container
        # MuSc accumulates results over r_list
        anomaly_maps_r = torch.tensor([]).double() # Will hold (len(r_list), N, H*W)
        
        # Helper to get feature dim from first batch, first layer
        feature_dim = patch_tokens_list[0][0].shape[-1]
        
        for r in self.r_list:
            print(f'Processing aggregation degree r={r}...')
            LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)
            
            # Key difference: MuSc iterates simply: for im in range(len(patch_tokens_list)):
            # Warning: MuSc's patch_tokens_list variable seems to be per-batch list in the loop
            # "patch_tokens_list.append(patch_tokens)" -> List of List of Tensors(Batch)
            
            Z_layers = {} # dict of list of tensors
            
            # Aggregate features for all images using LNAMD
            for batch_idx, batch_feats in enumerate(patch_tokens_list):
                # batch_feats is [Layer1(B,L,C), Layer2(B,L,C)...]
                
                # LNAMD expects input as list of tensors, but usually for a single image? 
                # Let's check LNAMD._embed. It takes `features_list`.
                # In MuSc.py: LNAMD_r._embed(patch_tokens) where patch_tokens is from patch_tokens_list[im]
                # It seems LNAMD can handle batch dimension if B dim exists.
                
                # Move batch to device
                batch_feats_gpu = [f.to(self.device) for f in batch_feats]
                
                with torch.no_grad():
                     # LNAMD._embed returns (B, N_patches, num_layers, C_out)
                    features = LNAMD_r._embed(batch_feats_gpu)
                    features /= features.norm(dim=-1, keepdim=True)
                    
                    for l_idx in range(len(self.features_list)):
                        l_key = str(l_idx)
                        if l_key not in Z_layers: Z_layers[l_key] = []
                        # features[:, :, l_idx, :] shape is (B, N_patches, C_out)
                        Z_layers[l_key].append(features[:, :, l_idx, :])
            
            # MSM Scoring
            anomaly_maps_l = torch.tensor([]).double() # (N, L) i.e. (Total_Img, N_patches)
            
            for l_key in Z_layers.keys():
                # Concatenate all batches for this layer -> (Total_Images, N_patches, C)
                Z = torch.cat(Z_layers[l_key], dim=0).to(self.device)
                
                # MSM
                # Returns (Total_Images, N_patches)
                scores = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
                
                # Accumulate layers (MuSc uses torch.cat then mean)
                anomaly_maps_l = torch.cat((anomaly_maps_l, scores.unsqueeze(0).cpu()), dim=0)
                del Z
                torch.cuda.empty_cache()
            
            # Mean over layers
            anomaly_maps_l = torch.mean(anomaly_maps_l, dim=0) # (Total_Images, N_patches)
            anomaly_maps_r = torch.cat((anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0)

        # Mean over r_list
        # anomaly_maps_iter: (Total_Images, N_patches) (on GPU)
        anomaly_maps_final = torch.mean(anomaly_maps_r, dim=0).to(self.device)
        
        # Interpolate and Save
        # Reshape to (N, 1, H, H)
        N, L_patches = anomaly_maps_final.shape
        H_feat = int(np.sqrt(L_patches))
        
        # Interpolate to original image size
        anomaly_maps_resized = F.interpolate(
            anomaly_maps_final.view(N, 1, H_feat, H_feat),
            size=self.image_size, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Save results
        print(f"[MuScWrapper] Saving maps to {output_dir}")
        saved_paths = []
        maps_np = anomaly_maps_resized.cpu().numpy() # (N, 1, H, W)
        
        for i, img_path in enumerate(image_paths_all):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{base_name}_map.npy")
            
            # Save the single map (H, W)
            map_data = maps_np[i, 0, :, :]
            np.save(save_path, map_data)
            saved_paths.append(save_path)
            
        print(f"[MuScWrapper] Generated {len(saved_paths)} maps.")
        return saved_paths

