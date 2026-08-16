
# Go to the deployment folder and run the following command to start the Streamlit app:
# streamlit run app.py


import json
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "brain_tumor_model.pth"
CLASS_NAMES_PATH = "class_names.json"
# ============================================================
# Streamlit Page Configuration
# ============================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)
# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    # Load class names
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    # EfficientNet-B0
    model = models.efficientnet_b0(weights=None)
    # Recreate classifier exactly as training
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.15),
        nn.Linear(256, len(class_names))
    )
    # Load trained weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model, class_names
# Load model
model, class_names = load_model()
# ============================================================
# Image Preprocessing
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

# ============================================================
# User Interface
# ============================================================
st.title("🧠 Brain Tumor Detection from MRI")
st.write(
    "Upload a brain MRI image to classify the tumor type "
    "and visualize the model's attention using Grad-CAM."
)
uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["png", "jpg", "jpeg"]
)

# ============================================================
# Prediction
# ============================================================
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    # Display uploaded image
    st.subheader("Uploaded MRI")
    st.image(
        image,
        caption="MRI Image",
        use_container_width=True
    )
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        confidence = float(probs[pred_idx])

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------
    try:
        # Grad-CAM requires gradients
        model.eval()

        target_layers = [
            model.features[-1]
        ]

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        targets = [
            ClassifierOutputTarget(pred_idx)
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

        # Resize original image to 224x224
        rgb_img = np.array(
            image.resize(
                (IMG_SIZE, IMG_SIZE)
            )
        ).astype(np.float32) / 255.0
        # Generate heatmap
        heatmap = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        # Display Grad-CAM
        st.subheader("Grad-CAM Visualization")
        st.image(
            heatmap,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    except Exception as e:
        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

    # --------------------------------------------------------
    # Prediction Result
    # --------------------------------------------------------
    st.subheader("Prediction Result")
    st.success(
        f"Prediction: {class_names[pred_idx]}"
    )
    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Class Probabilities
    # --------------------------------------------------------

    st.subheader("Class Probabilities")

    probability_data = {
        class_names[i]: float(probs[i])
        for i in range(len(class_names))
    }

    st.bar_chart(probability_data)

