# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob
import numpy as np
import cv2
import torch
import warnings
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineEngine")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from core.musc_wrapper import MuScWrapper
    from core.AnomalyNCD_wrapper import AnomalyNCDWrapper
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error("Please ensure your PYTHONPATH includes the project root.")

class PipelineEngine:
    def __init__(self, output_base_dir=None):
        self.project_root = PROJECT_ROOT
        self.libs_path = os.path.join(self.project_root, 'libs')
        
        # Paths to default configs
        self.musc_config = os.path.join(self.libs_path, 'MuSc', 'configs', 'musc.yaml')
        self.ncd_config = os.path.join(self.libs_path, 'AnomalyNCD', 'configs', 'AnomalyNCD.yaml')
        
        # Output setup
        if output_base_dir is None:
            self.output_base = os.path.join(self.project_root, 'data_store', 'results')
        else:
            self.output_base = output_base_dir
            
    def _convert_npy_to_png(self, npy_path, out_dir):
        """
        Convert single npy map to png.
        """
        try:
            map_data = np.load(npy_path)
            
            # Normalize to 0-1 range for visualization/NCD processing
            if map_data.max() - map_data.min() > 1e-9:
                norm_map = (map_data - map_data.min()) / (map_data.max() - map_data.min())
            else:
                norm_map = map_data
            
            # Convert to 8-bit [0, 255]
            img_map = (norm_map * 255).astype(np.uint8)
            
            # Clean filename
            base_name = os.path.splitext(os.path.basename(npy_path))[0]
            clean_name = base_name.replace('_map', '')
             
            out_path = os.path.join(out_dir, clean_name + '.png')
            cv2.imwrite(out_path, img_map)
            return True
        except Exception as e:
            logger.error(f"Error converting {npy_path}: {e}")
            return False

    def bridge_maps(self, npy_dir, png_dir):
        """
        Batch conversion from MuSc .npy output to AnomalyNCD .png input.
        """
        os.makedirs(png_dir, exist_ok=True)
        npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
        logger.info(f"Bridging {len(npy_files)} maps...")
        
        count = 0
        for f in npy_files:
            if self._convert_npy_to_png(f, png_dir):
                count += 1
        logger.info(f"Successfully bridged {count} maps.")
        return png_dir

    def check_aux_data(self):
        """
        Ensure auxiliary data (AeBAD_crop) exists for AnomalyNCD to function.
        """
        aux_data_path = os.path.join(self.libs_path, 'AnomalyNCD', 'data', 'AeBAD_crop', 'images')
        # Check parent folder existence at least
        if not os.path.exists(aux_data_path):
             os.makedirs(aux_data_path, exist_ok=True)
             
        if not os.listdir(aux_data_path):
            logger.warning(f"Auxiliary dataset not found at {aux_data_path}")
            logger.warning("Creating dummy auxiliary data to prevent NCD crash...")
            
            dummy_class_dir = os.path.join(aux_data_path, 'dummy_class')
            os.makedirs(dummy_class_dir, exist_ok=True)
            
            # Create a few dummy black images
            for i in range(5):
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(dummy_class_dir, f'{i:03d}.png'), dummy_img)
            logger.info("Dummy auxiliary data created.")

    def run_pipeline(self, input_dir, run_name=None):
        """
        Main execution flow.
        """
        # Timestamp for unique run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder_name = f"Run_{timestamp}" if run_name is None else f"{run_name}_{timestamp}"
        
        session_dir = os.path.join(self.output_base, run_folder_name)
        os.makedirs(session_dir, exist_ok=True)
        
        musc_out_dir = os.path.join(session_dir, 'musc_maps_npy')
        bridge_out_dir = os.path.join(session_dir, 'musc_maps_png')
        ncd_out_dir = os.path.join(session_dir, 'ncd_results')
        
        logger.info("="*50)
        logger.info(f"STARTING PIPELINE SESSION: {run_folder_name}")
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {session_dir}")
        logger.info("="*50)
        
        # --- Stage 1: MuSc Generator ---
        logger.info(">>> Stage 1: Running MuSc (Anomaly Map Generator)...")
        try:
            musc_wrapper = MuScWrapper(self.musc_config) # Initialize MuSc
            
            if not os.path.exists(input_dir) or not os.listdir(input_dir):
                 logger.error(f"Input directory {input_dir} is empty or missing!")
                 return False

            generated_paths = musc_wrapper.generate_anomaly_maps(input_dir, musc_out_dir)
            
            # Force cleanup to free VRAM for next stage
            del musc_wrapper
            torch.cuda.empty_cache()
            
            if not generated_paths:
                logger.error("MuSc failed to generate any maps.")
                return False
                
            logger.info(f"Stage 1 Complete. Maps saved to {musc_out_dir}")
            
        except Exception as e:
            logger.critical(f"Stage 1 Failed: {e}", exc_info=True)
            return False

        # --- Stage 2: Data Bridge ---
        logger.info(">>> Stage 2: Bridging Data Formats...")
        try:
             self.bridge_maps(musc_out_dir, bridge_out_dir)
             logger.info("Stage 2 Complete.")
        except Exception as e:
             logger.critical(f"Stage 2 Failed: {e}", exc_info=True)
             return False

        # --- Stage 3: AnomalyNCD Discovery ---
        logger.info(">>> Stage 3: Running AnomalyNCD (Novel Class Discovery)...")
        try:
            self.check_aux_data()
            
            ncd_wrapper = AnomalyNCDWrapper(self.ncd_config)
            
            # Run NCD
            # Note: inputs are input_dir (original images) and bridge_out_dir (png maps)
            results = ncd_wrapper.run(input_dir, bridge_out_dir, ncd_out_dir)
            
            logger.info("Stage 3 Complete.")
            
            # Save results summary
            summary_file = os.path.join(session_dir, 'pipeline_summary.json')
            with open(summary_file, 'w') as f:
                json.dump({
                    "success": True, 
                    "run_id": run_folder_name,
                    "stages": ["MuSc", "Bridge", "NCD"],
                    "results_path": ncd_out_dir,
                    "metrics": str(results)
                }, f, indent=4)
                
            logger.info("="*50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Final results are in: {ncd_out_dir}")
            logger.info("="*50)
            return True
            
        except Exception as e:
            logger.critical(f"Stage 3 Failed: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DCProject2026 Core Engine")
    parser.add_argument('--input', type=str, required=True, help="Path to input images folder")
    parser.add_argument('--output', type=str, default=None, help="Base path for results")
    parser.add_argument('--name', type=str, default=None, help="Optional name for this run")
    
    args = parser.parse_args()
    
    engine = PipelineEngine(output_base_dir=args.output)
    engine.run_pipeline(args.input, run_name=args.name)
