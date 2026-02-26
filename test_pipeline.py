# -*- coding: utf-8 -*-
import os
import sys
import shutil
import numpy as np
import cv2
import torch

# Dynamically add paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'core'))

from core.musc_wrapper import MuScWrapper
from core.data_bridge import DataBridge

def setup_test_environment():
    """
    Create an isolated test environment and generate dummy images to test the pipeline.
    """
    print("=== Setting up Test Environment ===")
    
    # 1. Define test directories
    test_dir = os.path.join(PROJECT_ROOT, 'test_workspace')
    raw_dir = os.path.join(test_dir, 'raw_inputs')
    maps_dir = os.path.join(test_dir, 'intermediate', 'maps')
    ncd_dir = os.path.join(test_dir, 'intermediate', 'ncd_data')
    
    # Clean up old test directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(ncd_dir, exist_ok=True)
    
    # 2. Generate 3 dummy images (solid background with a random "anomaly" block)
    print(f"Generating dummy images in {raw_dir}...")
    for i in range(3):
        # Create 224x224 gray background
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Draw a random black rectangle as "anomaly"
        x, y = np.random.randint(20, 180, size=2)
        w, h = np.random.randint(20, 50, size=2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)
        
        cv2.imwrite(os.path.join(raw_dir, f"dummy_test_{i}.jpg"), img)
        
    print("Test environment ready.\n")
    return raw_dir, maps_dir, ncd_dir, test_dir

def run_test_pipeline():
    """
    Run the test pipeline: MuSc -> DataBridge
    """
    raw_dir, maps_dir, ncd_dir, test_dir = setup_test_environment()
    
    print("=== Stage 1: Testing MuScWrapper ===")
    # Note: This requires a real musc.yaml config file path
    # If libs/MuSc/configs/musc.yaml does not exist, it will fail here
    config_path = os.path.join(PROJECT_ROOT, 'libs', 'MuSc', 'configs', 'musc.yaml')
    
    if not os.path.exists(config_path):
        print(f"? Error: Config file not found at {config_path}")
        print("Please ensure the MuSc library is correctly placed.")
        return
        
    try:
        # Instantiate Wrapper (using GPU if available, otherwise CPU)
        # Note: Pass integer 0 for cuda:0, or string 'cpu'. MuSc likely expects an int for GPU index.
        device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        musc = MuScWrapper(config_path=config_path, device=device)
        
        # Run generation
        saved_maps = musc.generate_anomaly_maps(raw_dir, maps_dir)
        
        if len(saved_maps) == 3:
            print("? Stage 1 (MuSc) Passed! 3 maps generated.")
        else:
            print(f"? Stage 1 Failed. Expected 3 maps, got {len(saved_maps)}.")
            return
            
    except Exception as e:
        print(f"? Stage 1 (MuSc) Crashed with error:\n{e}")
        return

    print("\n=== Stage 2: Testing DataBridge ===")
    try:
        bridge = DataBridge(anomaly_thred=0.5)
        
        # Run bridge
        final_dataset_dir = bridge.prepare_ncd_dataset(raw_dir, maps_dir, ncd_dir)
        
        # Verify output directory structure
        check_img_dir = os.path.join(final_dataset_dir, "unknown_batch", "images")
        check_mask_dir = os.path.join(final_dataset_dir, "unknown_batch", "masks")
        
        if os.path.exists(check_img_dir) and len(os.listdir(check_img_dir)) == 3:
            if os.path.exists(check_mask_dir) and len(os.listdir(check_mask_dir)) == 3:
                print("? Stage 2 (DataBridge) Passed! Images and Masks are ready.")
                print(f"? Pipeline Test Successful! Check the results in: {test_dir}")
            else:
                print("? Stage 2 Failed. Masks not generated correctly.")
        else:
            print("? Stage 2 Failed. Images not copied correctly.")
            
    except Exception as e:
        print(f"? Stage 2 (DataBridge) Crashed with error:\n{e}")

if __name__ == "__main__":
    run_test_pipeline()
