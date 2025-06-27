import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the trained model
MODEL_PATH = './models/best.pt'


def process_image(image):
    model = YOLO(MODEL_PATH)
    result = model(image)[0]
    annotated_image = result.plot()  # Plot the bounding boxes on the image
    annotated_image = Image.fromarray(annotated_image)
    return annotated_image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        # Display the uploaded image
        uploaded_image = Image.open(file)

        # Process the image
        image = np.array(uploaded_image)
        annotated_image = process_image(image)

        # Display the images side by side
        if annotated_image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_image, caption="Uploaded Image")
            with col2:
                st.image(annotated_image, caption="Predicted Image")
        else:
            st.error(
                "Prediction image not found. Please check the process_image function.")


if __name__ == "__main__":
    main()
