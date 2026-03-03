# -*- coding: utf-8 -*-
import os
import sys
import argparse
import shutil
import json
import cv2
import numpy as np
import torch
import warnings
from types import SimpleNamespace

# Dynamically add libs path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANOMALYNCD_PATH = os.path.join(PROJECT_ROOT, 'libs', 'AnomalyNCD')
sys.path.append(ANOMALYNCD_PATH)

from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml
from models.modules._MEBin import MEBin
from models.loss._distill_loss import DistillLoss
from models.modules._classifier import get_params_groups
from torch.optim import SGD, lr_scheduler

warnings.filterwarnings("ignore")

class AnomalyNCDWrapper(AnomalyNCD):
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(ANOMALYNCD_PATH, 'configs', 'AnomalyNCD.yaml')
        
        # Load config
        self.cfg = load_yaml(config_path)
        
        # Initialize args with defaults from config
        self.args = self._init_args(self.cfg)
        
        # Call super init (stores args)
        super().__init__(self.args)
        
        # Ensure device is set (usually done in train_init, but good to have)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _init_args(self, cfg):
        args = SimpleNamespace()
        
        # Default args matching anomalyncd_main.py & requirements
        args.dataset = 'custom'  
        args.category = 'default' # Will be used as folder name
        args.dataset_path = None
        args.anomaly_map_path = None
        args.binary_data_path = None
        args.crop_data_path = None
        
        # Set base_data_path to the one requested by user
        # User specified: libs/AnomalyNCD/data/AeBAD_crop
        args.base_data_path = os.path.join(ANOMALYNCD_PATH, 'data', 'AeBAD_crop')
        
        args.config = None
        args.runner_name = 'AnomalyNCD_Wrapped'
        args.only_test = None
        args.checkpoint_path = None

        # Load from cfg
        # binarization
        args.sample_rate = cfg['binarization']['sample_rate']
        args.min_interval_len = cfg['binarization']['min_interval_len']
        args.erode = cfg['binarization']['erode']
        # model
        args.grad_from_block = cfg['models']['grad_from_block']
        args.pretrained_backbone = cfg['models']['pretrained_backbone']
        args.mask_layers = cfg['models']['mask_layers']
        args.n_views = cfg['models']['n_views']
        args.n_head = cfg['models']['n_head']
        # training
        args.batch_size = cfg['training']['batch_size']
        args.num_workers = cfg['training']['num_workers']
        args.lr = cfg['training']['lr']
        args.gamma = cfg['training']['gamma']
        args.momentum = cfg['training']['momentum']
        args.weight_decay = cfg['training']['weight_decay']
        args.epochs = cfg['training']['epochs']
        # loss
        args.sup_weight = cfg['loss']['sup_weight']
        args.memax_weight = cfg['loss']['memax_weight']
        args.anomaly_thred = cfg['loss']['anomaly_thred']
        args.teacher_temp = cfg['loss']['teacher_temp']
        args.warmup_teacher_temp = cfg['loss']['warmup_teacher_temp']
        args.warmup_teacher_temp_epochs = cfg['loss']['warmup_teacher_temp_epochs']
        args.repeat_times = cfg['loss']['repeat_times']
        # experiment
        args.seed = cfg['experiment']['seed']
        args.print_freq = cfg['experiment']['print_freq']
        args.table_root = cfg['experiment']['table_root']
        args.exp_name = cfg['experiment']['exp_name']
        args.exp_root = cfg['experiment']['exp_root']
        
        # Initialize logger as AnomalyNCD expects it
        import logging
        args.logger = logging.getLogger(args.runner_name)
        if not args.logger.handlers:
             ch = logging.StreamHandler()
             ch.setLevel(logging.INFO)
             args.logger.addHandler(ch)
             args.logger.setLevel(logging.INFO)

        return args

    def binarization_wrapper(self, input_dir, maps_dir):
        """
        Custom binarization that handles flat directory input.
        Generates crops and masks needed for NCD training.
        """
        print(f"Binarizing data from {input_dir} using maps from {maps_dir}...")
        self.args.dataset_path = input_dir
        self.args.anomaly_map_path = maps_dir
        
        product_name = self.args.category
        output_path = self.args.binary_data_path
        crop_output_path = self.args.crop_data_path
        
        # Ensure output directories exist
        if not os.path.exists(output_path):
             os.makedirs(output_path)
        if not os.path.exists(crop_output_path):
             os.makedirs(crop_output_path)

        # Clean up category dirs
        for path in [output_path, crop_output_path]:
            ppath = os.path.join(path, product_name)
            if os.path.exists(ppath):
                shutil.rmtree(ppath)
            os.makedirs(ppath, exist_ok=True)
            
        # Collect files - assume flat structure for "custom" inputs from pipeline
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        img_files = sorted([
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(valid_exts)
        ])
        
        if not img_files:
            raise ValueError(f"No valid images found in {input_dir}")

        map_files = []
        valid_img_files = []
        for img_path in img_files:
            fname = os.path.basename(img_path)
            
            # Map search strategy:
            # 1. Exact match
            map_path = os.path.join(maps_dir, fname)
            
            # 2. Try png extension if original is different
            if not os.path.exists(map_path):
                 base, _ = os.path.splitext(fname)
                 map_path = os.path.join(maps_dir, base + '.png')
            
            if os.path.exists(map_path):
                valid_img_files.append(img_path)
                map_files.append(map_path)
            else:
                print(f"Warning: Map not found for {img_path}, skipping.")

        if not valid_img_files:
            raise ValueError("No valid image-map pairs found.")

        # Treat all inputs as a single "anomaly_type" -> "data"
        # This structure is required by AnomalyNCD's dataset loader
        anomaly_type = "data"
        
        # Run MEBin
        bin_tool = MEBin(self.args, map_files)
        # binarize_anomaly_maps() returns list of numpy arrays (0/255)
        binarized_maps_list, est_anomaly_nums_list = bin_tool.binarize_anomaly_maps()
        
        # Save binarized maps
        anomaly_type_out_path = os.path.join(output_path, product_name, anomaly_type)
        os.makedirs(anomaly_type_out_path, exist_ok=True)
        
        for i, binarized_map in enumerate(binarized_maps_list):
            # Save using map filename
            map_fname = os.path.basename(map_files[i])
            save_map_path = os.path.join(anomaly_type_out_path, map_fname)
            
            # Resize logic from original code (ensure mask matches image size)
            o_img = cv2.imread(valid_img_files[i])
            o_img_shape = o_img.shape
            # binarized_map is grayscale height x width
            binarized_map = cv2.resize(binarized_map, (o_img_shape[1], o_img_shape[0]), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(save_map_path, binarized_map)

        # Crop and save Sub-images
        # Structure: crop_data_path/category/images/anomaly_type/
        save_path = os.path.join(crop_output_path, product_name, 'images', anomaly_type)
        save_mask_path = os.path.join(crop_output_path, product_name, 'masks', anomaly_type)
        
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_mask_path, exist_ok=True)
        
        id_score_list = {}
        
        print(f"Cropping {len(valid_img_files)} images...")
        for i in range(len(valid_img_files)):
            image_path = valid_img_files[i]
            map_path = map_files[i]
            
            map_fname = os.path.basename(map_path)
            binary_map_path = os.path.join(anomaly_type_out_path, map_fname)
            
            est_ano_num = est_anomaly_nums_list[i]
            
            anomaly_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            binary_map = cv2.imread(binary_map_path, cv2.IMREAD_GRAYSCALE) 
            
            # crop_sub_image_mask returns PIL images
            # Must convert them to 'RGB' or 'L' before saving if they are not? 
            # Original code saves them as png.
            
            sub_images_list, sub_masks_list, anomaly_crop_score = bin_tool.crop_sub_image_mask(
                image=image, mask=binary_map, anomaly_map=anomaly_map, est_anomaly_num=est_ano_num
            )
            
            prefix = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save crops
            for idx, img_crop in enumerate(sub_images_list):
                img_crop.save(os.path.join(save_path, "{}_crop{}.png".format(prefix, idx)))
            for idx, mask_crop in enumerate(sub_masks_list):
                mask_crop.save(os.path.join(save_mask_path, "{}_crop{}.png".format(prefix, idx)))
            
            # Record scores
            for idx, score in enumerate(anomaly_crop_score):
                id_score_list["{}_crop{}.png".format(prefix, idx)] = score / 255.0
        
        # Save scores json required by MGRL
        full_score_list = {anomaly_type: id_score_list}
        json_dir = os.path.join(crop_output_path, 'scores_json')
        os.makedirs(json_dir, exist_ok=True)
        with open(os.path.join(json_dir, f'{product_name}.json'), 'w') as f:
            json.dump(full_score_list, f)
            
        print(f"Binarization complete. Crops saved to {crop_output_path}")

    def run(self, input_dir, maps_dir, output_dir):
        """
        Main execution method.
        Args:
            input_dir: Directory containing input images
            maps_dir: Directory containing anomaly maps
            output_dir: Directory to save results (binary masks, crops, model logs)
        """
        # Ensure output directories exist with absolute paths
        output_dir = os.path.abspath(output_dir)
        input_dir = os.path.abspath(input_dir)
        maps_dir = os.path.abspath(maps_dir)
        
        self.args.binary_data_path = os.path.join(output_dir, 'binary_masks')
        self.args.crop_data_path = os.path.join(output_dir, 'crops')
        self.args.exp_root = os.path.join(output_dir, 'exp')
        os.makedirs(self.args.exp_root, exist_ok=True)
        
        # 1. Binarization and Cropping 
        # (This adapts the folder input to NCD's expected structure)
        self.binarization_wrapper(input_dir, maps_dir)
        
        # 2. Init Training
        # This will load datasets using the crops generated above
        print("Initializing Training...")
        self.train_init()
        
        # 3. Model Training / Discovery Loop
        print("Prepare optimizer...")
        params_groups = get_params_groups(self.model)
        optimizer = SGD(params_groups, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.lr * 1e-3,
            )

        cluster_criterion = DistillLoss(
                            warmup_teacher_temp_epochs=self.args.warmup_teacher_temp_epochs,
                            nepochs=self.args.epochs,
                            ncrops=self.args.n_views,
                            warmup_teacher_temp=self.args.warmup_teacher_temp,
                            teacher_temp=self.args.teacher_temp,
                            num_labeled_classes=self.args.num_labeled_classes,
                            num_unlabeled_classes=self.args.num_unlabeled_classes,
                            student_temp=0.1,
                            repeat_times=self.args.repeat_times
                        ).cuda()

        print(f"Starting Training for {self.args.epochs} epochs...")
        
        results_merge = None
        cluster_loss_head = []
        
        for epoch in range(self.args.epochs):
            cluster_loss_head, loss_record = self.MGRL(epoch, optimizer, cluster_criterion)
            
            if epoch % self.args.print_freq == 0:
                print('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
            
            exp_lr_scheduler.step()

        # Final Save & Evaluation
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': self.args.epochs,
            'loss_list': cluster_loss_head,
            'base_category': self.args.base_category,
            'category': self.args.category,
            'mask_layers': self.args.mask_layers
        }
        model_save_path = os.path.join(self.args.exp_root, f'model_{self.args.category}.pth')
        torch.save(save_dict, model_save_path)
        print(f"Model saved to {model_save_path}")

        print('Predicting for Sub-Image Classification...')
        self.sub_image_predict(epoch=self.args.epochs, save_name='Sub-image prediction', loss_list=cluster_loss_head)
        
        print('Region Merging for Image Classification...')
        results_merge, _ = self.region_merge_predict(epoch=self.args.epochs, save_name='Region merged prediction', loss_list=cluster_loss_head)
                
        return results_merge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AnomalyNCD Wrapper')
    parser.add_argument('--input', type=str, required=True, help='Input images folder')
    parser.add_argument('--maps', type=str, required=True, help='Input anomaly maps folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    
    args = parser.parse_args()
    
    wrapper = AnomalyNCDWrapper()
    wrapper.run(args.input, args.maps, args.output)
