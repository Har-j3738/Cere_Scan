"""CereScan Streamlit application."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from neuro_scan.model import CLASS_NAMES, load_checkpoint

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cerescan_resnet18.pth"

st.set_page_config(page_title="CereScan", page_icon="🧠", layout="wide")

st.title("🧠 CereScan")
st.subheader("AI-assisted brain MRI classification")
st.caption("Research / educational prototype")

st.sidebar.title("CereScan")
st.sidebar.write("Real PyTorch inference • ResNet18")
st.sidebar.warning("Not a medical diagnostic system.")

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_checkpoint(MODEL_PATH, device), device

def preprocess(image):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])(image.convert("RGB")).unsqueeze(0)

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if not MODEL_PATH.exists():
    st.info("The trained model is not present yet. Prepare the dataset and train CereScan first.")
elif uploaded:
    image = Image.open(uploaded)
    left, right = st.columns(2)

    with left:
        st.subheader("MRI Scan")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Analysis")
        if st.button("🔬 Analyze MRI", type="primary"):
            model, device = get_model()
            tensor = preprocess(image).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

            predicted_index = int(probabilities.argmax())
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index])

            st.success("Inference completed")
            st.metric("Predicted Class", predicted_class)
            st.metric("Model Probability", f"{confidence * 100:.1f}%")

            st.subheader("Class probabilities")
            for label, probability in zip(CLASS_NAMES, probabilities.tolist()):
                st.progress(float(probability), text=f"{label}: {probability * 100:.1f}%")

            st.warning(
                "The probability shown is the neural network's output, not a "
                "clinical certainty or diagnosis."
            )
else:
    st.info("Upload an MRI image to begin.")

st.divider()
st.caption(
    "CereScan is an educational/research prototype. Do not use it to "
    "diagnose, treat, or make clinical decisions."
)
