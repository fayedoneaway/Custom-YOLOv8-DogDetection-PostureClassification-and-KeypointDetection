import streamlit as st
import os

st.title(
    "Custom YOLOv8 Model:\n"
    "Dog Detection with Posture Classification\n"
    "and Keypoint Detection"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

images = {
    "Image 1": ("Image1.jpg", "pred_Image1.jpg"),
    "Image 2": ("Image2.jpg", "pred_Image2.jpg"),
    "Image 3": ("Image3.jpg", "pred_Image3.jpg"),
    "Image 4": ("Image4.jpg", "pred_Image4.jpg"),
}

choice = st.selectbox(
    "Choose an image", list(images.keys())
)

raw_file, pred_file = images[choice]

raw_path = os.path.join(BASE_DIR, raw_file)
pred_path = os.path.join(BASE_DIR, pred_file)

if not os.path.exists(raw_path):
    st.error(f"Raw image not found: {raw_path}")
    st.stop()

if not os.path.exists(pred_path):
    st.error(f"Prediction image not found: {pred_path}")
    st.stop()


st.subheader(r"Raw Image:")
st.image(raw_path, caption="Raw Image", use_container_width=True)

st.subheader(
    "Prediction: Dog Posture and Dog Attributes \n"
    "Nose, Left Ear, Right Ear, Left Eye, Right Eye"
)

st.image(
    pred_path,
    caption="Posture and Attributes Prediction",
    use_container_width=True
)