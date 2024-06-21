# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from streamlit_modal import Modal
from collections import Counter


# Load YOLOv8 model
@st.cache_data
def load_model(weights='yolov8n.pt'):
    model = YOLO(weights)
    return model

# method that preprocess input images to match YOLO model input
def preprocess_image(image):
    image = image.resize((640, 640))  # 1-resize the input image to match the model input shape
    image = np.array(image)    # 2- convert the input image format to numpy array
    # image = image.astype(np.float32) / 255.0   # 3- normalize the array values
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

# create an instance of YOLOv8 model
model = load_model()

# Setup streamlit page
st.title("Object Detection")
st.write("Upload an image to detect objects")

# set file uploader to allow images with extensions (jpeg,jpg,png,bmp)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","bmp"])

if uploaded_file is not None:

    # open the uploaded image and display it
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # add the "Analyse Image" button in the center
    col1, col2, col3 = st.columns([1.8, 1.5, 1.2])
    with col2:
        analyse_button = st.button("Analyse Image") if uploaded_file is not None else st.button("Analyse Image", disabled=True)

     # when button is triggered
    if analyse_button:
        # 1- Preprocess the input image
        processed_image = preprocess_image(image)

        # 2- Get results from the yolo model on the preprocessed img
        results = model(processed_image)

        # get the detected objects from the model results
        detections = results[0]

        if detections.boxes is not None:

            # Get the detected class label for each object and map it to corresponding name
            class_indices = detections.boxes.cls.cpu().numpy().astype(int)  # 1-  Get class indices
            class_names = [model.names[i] for i in class_indices]  # 2- Map indices to class names
            counts = {name: class_names.count(name) for name in set(class_names)}  # 3-Count occurrences of each class

            # Format the results for display
            display_text = "\n\n".join([f"{obj} : {count}" for obj, count in counts.items()])

            # Creating modal container for popping up the result
            modal = Modal(
                key="demo-modal",
                title="This Image Contains : ",

                # Optional
                padding=60,  # default value
                max_width=500  # default value
            )
            # Define the content of the modal
            with modal.container():
                # Display the content of the modal
                st.write(display_text)

                if st.button("OK", key="close-modal"):
                    modal.toggle()

        else:
            st.warning("No objects detected.")

