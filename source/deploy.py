import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(layout="wide")

# Load the trained model
MODEL_PATH = './models/best.pt'


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Process the uploaded image with the model


def process_image(image, model):
    result = model(image)[0]
    annotated_image = Image.fromarray(result.plot())
    return annotated_image


def main():
    st.title('Object Detection for Images')

    file = st.sidebar.file_uploader(
        'Upload Image', type=['jpg', 'png', 'jpeg'])

    if file is not None:
        try:
            uploaded_image = Image.open(file).convert("RGB")
            image = np.array(uploaded_image)
        except Exception as e:
            st.error(f"❌ Failed to load image: {e}")
            return

        model = load_model()
        with st.spinner("Running detection..."):
            try:
                annotated_image = process_image(image, model)
            except Exception as e:
                st.error(f"❌ Model prediction failed: {e}")
                return
        with st.container():
            st.write("## Results")
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col2:
                st.image(uploaded_image, caption="Uploaded Image",
                         use_column_width=True)
            with col3:
                st.image(annotated_image, caption="Predicted Image",
                         use_column_width=True)


if __name__ == "__main__":
    main()
