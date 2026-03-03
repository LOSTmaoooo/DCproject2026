# -*- coding: utf-8 -*-
import os
import sys
import shutil
import numpy as np
import cv2
import argparse

# Dynamically add paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from core.engine import PipelineEngine

def setup_test_environment(num_images=5):
    """
    Create an isolated test environment and generate dummy images.
    """
    print("=== Setting up Test Environment ===")
    
    # Define test directories
    test_dir = os.path.join(PROJECT_ROOT, 'test_workspace')
    raw_dir = os.path.join(test_dir, 'raw_inputs')
    
    # Clean up old test directory
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
        except Exception as e:
            print(f"Warning: Could not delete old test directory. Proceeding anyway. {e}")
            
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Generating {num_images} dummy images in {raw_dir}...")
    for i in range(num_images):
        # Create 224x224 gray background
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Add some random noise
        noise = np.random.randint(0, 20, (224, 224, 3)).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add "anomaly" block
        x = np.random.randint(20, 150)
        y = np.random.randint(20, 150)
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), -1)
        
        # Save
        cv2.imwrite(os.path.join(raw_dir, f"test_sample_{i:03d}.png"), img)
        
    print("Test environment ready.\n")
    return test_dir, raw_dir

def main():
    parser = argparse.ArgumentParser(description="Test Pipeline Runner")
    parser.add_argument('--setup_only', action='store_true', help="Only setup test data")
    args = parser.parse_args()

    # 1. Setup Data
    test_dir, raw_dir = setup_test_environment()
    
    if args.setup_only:
        print(f"Setup complete. Data is in {raw_dir}")
        return

    # 2. Initialize Engine
    print("=== Initializing Engine ===")
    
    # We set output directory to inside test_workspace
    results_dir = os.path.join(test_dir, 'results')
    
    # Instantiate Engine
    try:
        engine = PipelineEngine(output_base_dir=results_dir)
    except Exception as e:
        print(f"Failed to initialize Engine: {e}")
        sys.exit(1)
    
    # 3. Run Pipeline
    print("=== Running Pipeline ===")
    
    # Run synchronously
    try:
        success = engine.run_pipeline(input_dir=raw_dir, run_name="AutomatedTest")
        
        if success:
            print("\n>>> TEST PASSED: Pipeline completed successfully. <<<")
            print(f"Outputs located at: {results_dir}")
        else:
            print("\n>>> TEST FAILED: Pipeline encountered errors. <<<")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest Crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
