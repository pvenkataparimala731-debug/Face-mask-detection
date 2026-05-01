import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Page Config
st.set_page_config(page_title='Face Mask Detector', layout='centered')
st.title('😷 Real-Time Face Mask Detection')
st.markdown('Upload an image to detect if people are wearing masks correctly.')

# Load Model
@st.cache_resource
def load_model():
    drive_path = '/content/drive/MyDrive/Mask_Detection_Export/best.onnx'
    local_path = 'best.onnx'
    
    model_to_use = drive_path if os.path.exists(drive_path) else local_path
    
    if not os.path.exists(model_to_use):
        return None, model_to_use
    return ort.InferenceSession(model_to_use), model_to_use

session, path_used = load_model()

if session:
    st.success(f'Model loaded successfully from: {path_used}')
else:
    st.error(f'Model file not found at {path_used}! Please ensure it exists.')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and session:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_resized = image.resize((320, 320))
    img_array = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    img_array = img_array[np.newaxis, :]

    # Inference
    if st.button('Detect Masks'):
        outputs = session.run(None, {'images': img_array})
        output_data = outputs[0][0]

        # Post-process (YOLOv8 parser)
        num_boxes = 2100
        conf_threshold = 0.3

        draw = np.array(image)
        h, w = draw.shape[:2]

        for i in range(num_boxes):
            mask_score = output_data[4][i]
            no_mask_score = output_data[5][i]
            score = max(mask_score, no_mask_score)

            if score > conf_threshold:
                cls_id = 0 if mask_score > no_mask_score else 1
                label = 'Mask' if cls_id == 0 else 'No Mask'
                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)

                cx, cy, bw, bh = output_data[0:4, i]
                x1 = int((cx - bw/2) * w / 320)
                y1 = int((cy - bh/2) * h / 320)
                x2 = int((cx + bw/2) * w / 320)
                y2 = int((cy + bh/2) * h / 320)

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f'{label} {score:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        st.image(draw, caption='Processed Image', use_column_width=True)
