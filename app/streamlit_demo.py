import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOUD_DIR = PROJECT_ROOT / 'cloud'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CLOUD_DIR) not in sys.path:
    sys.path.insert(0, str(CLOUD_DIR))

import api_client

st.set_page_config(page_title='Industrial Anomaly Detection System', layout='wide', initial_sidebar_state='expanded')

st.title('Industrial Anomaly Detection System')
st.caption('Local Streamlit demo connected to the shared backend API.')

mode = st.sidebar.radio('Select Operation Mode', ('Mode 1: Model Training & Registration', 'Mode 2: Single Stream Inference', 'Mode 3: Batch Analysis (Focus)'))
st.sidebar.caption(f'API Base URL: {os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")}')


def _save_upload(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return file_path


if mode == 'Mode 1: Model Training & Registration':
    st.header('Mode 1: Training & Setup')
    st.info('This mode currently registers part data and acknowledges the upload. Training remains a backend responsibility.')
    part_name = st.text_input('Part Name')
    uploaded_files = st.file_uploader('Upload images', accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])
    if st.button('Register Part'):
        if not part_name.strip():
            st.warning('Please enter a part name.')
        elif not uploaded_files:
            st.warning('Please upload at least one image.')
        else:
            tmp_dir = PROJECT_ROOT / 'data_store' / 'streamlit_uploads' / 'mode1'
            saved = [_save_upload(file, tmp_dir) for file in uploaded_files]
            with st.spinner('Submitting registration...'):
                result = api_client.register_part(part_name.strip(), saved)
            if result.get('code') == 200:
                st.success(result.get('msg', 'Registered successfully'))
                st.json(result.get('data'))
            else:
                st.error(result.get('msg', 'Request failed'))

elif mode == 'Mode 2: Single Stream Inference':
    st.header('Mode 2: Live Inference')
    model_options = api_client.fetch_model_list()
    model_name = st.selectbox('Select Model', [''] + [item['value'] for item in model_options], format_func=lambda x: 'Default model' if x == '' else next((item['label'] for item in model_options if item['value'] == x), x))
    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], accept_multiple_files=False)
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)
    if st.button('Run Inference'):
        if not uploaded_file:
            st.warning('Please upload an image first.')
        else:
            tmp_dir = PROJECT_ROOT / 'data_store' / 'streamlit_uploads' / 'mode2'
            image_path = _save_upload(uploaded_file, tmp_dir)
            with st.spinner('Running inference...'):
                result = api_client.predict_single(image_path, model_name or None)
            if result.get('code') == 200:
                data = result.get('data') or {}
                st.success(result.get('msg', 'Inference complete'))
                cols = st.columns(2)
                with cols[0]:
                    st.metric('Score', data.get('score', '--'))
                    st.metric('Cluster ID', data.get('clusterId', '--'))
                    st.metric('Anomaly Type', data.get('anomalyType', '--'))
                    st.metric('Latency', f"{data.get('latencyMs', '--')} ms")
                with cols[1]:
                    if data.get('imageUrl'):
                        st.image(data['imageUrl'], caption='Image URL from backend')
                    if data.get('heatmapUrl'):
                        st.image(data['heatmapUrl'], caption='Heatmap URL from backend')
                st.json(result)
            else:
                st.error(result.get('msg', 'Request failed'))

else:
    st.header('Mode 3: Batch Analysis')
    st.info('Upload a ZIP file and let the backend run MuSc first, then AnomalyNCD.')
    uploaded_zip = st.file_uploader('Upload ZIP package', type=['zip'])
    if st.button('Start Batch Pipeline'):
        if not uploaded_zip:
            st.warning('Please upload a ZIP file first.')
        else:
            tmp_dir = PROJECT_ROOT / 'data_store' / 'streamlit_uploads' / 'mode3'
            zip_path = _save_upload(uploaded_zip, tmp_dir)
            with st.spinner('Running batch analysis...'):
                result = api_client.run_batch_analysis(zip_path)
            if result.get('code') == 200:
                data = result.get('data') or {}
                st.success(result.get('msg', 'Batch analysis complete'))
                st.write(f"Result directory: {data.get('resultDir', '--')}")
                st.write(data.get('info', ''))
                st.balloons()
            else:
                st.error(result.get('msg', 'Request failed'))
