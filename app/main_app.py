import streamlit as st
import sys
import os

# Add project root to sys.path to allow imports from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("?? Industrial Anomaly Detection System")
st.markdown("---")

st.sidebar.title("System Mode")
mode = st.sidebar.radio(
    "Select Operation Mode:",
    ("Mode 1: Model Training & Registration", 
     "Mode 2: Single Stream Inference", 
     "Mode 3: Batch Analysis (Focus)")
)

if mode == "Mode 3: Batch Analysis (Focus)":
    st.header("? Mode 3: Batch Analysis")
    st.info("Upload a folder of images or Select a local directory to process a batch of images for Novel Class Discovery.")
    
    # Placeholder for batch processing UI
    input_dir = st.text_input("Input Directory Path:", value="d:/XYH/DCproject/data_store/raw_inputs")
    
    if st.button("Start Batch Pipeline"):
        st.write(f"? Starting analysis on: {input_dir}")
        from core.engine import BatchPipeline
        
        # Instantiate pipeline
        pipeline = BatchPipeline()
        
        with st.spinner('Running MuSc Generator...'):
            # Placeholder call
            pass
            
        with st.spinner('Running AnomalyNCD Analyzer...'):
             # Placeholder call
             pass
             
        st.success("Analysis Complete!")

elif mode == "Mode 1: Model Training & Registration":
    st.header("?? Mode 1: Training & Setup")
    st.warning("This mode is for registering new parts and training the classifier.")

elif mode == "Mode 2: Single Stream Inference":
    st.header("? Mode 2: Live Inference")
    st.warning("This mode is for real-time single image checking.")
