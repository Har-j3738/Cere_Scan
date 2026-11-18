"""CereScan Streamlit application."""

from pathlib import Path

import streamlit as st
from PIL import Image

from neuro_scan.inference import grad_cam, load_model, predict
from neuro_scan.visualization import overlay_heatmap

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cerescan_resnet18.pth"

st.set_page_config(page_title="CereScan", page_icon="🧠", layout="wide")

st.title("🧠 CereScan")
st.subheader("AI-assisted brain MRI classification")
st.caption("Research / educational prototype")

st.sidebar.title("CereScan")
st.sidebar.write("ResNet18 • real PyTorch inference • Grad-CAM")
st.sidebar.warning("Not a medical diagnostic system.")

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None, None
    return load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if not MODEL_PATH.exists():
    st.info("Model not found yet. Run dataset preparation and training first.")
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
            predicted_class, probability, probabilities = predict(model, image, device)

            st.success("Inference completed")
            st.metric("Predicted Class", predicted_class)
            st.metric("Model Probability", f"{probability * 100:.1f}%")

            st.subheader("Class probabilities")
            for label, value in zip(["Meningioma", "Glioma", "Pituitary"], probabilities):
                st.progress(float(value), text=f"{label}: {value * 100:.1f}%")

            st.subheader("AI explanation")
            st.caption("Grad-CAM highlights image regions that influenced the model output. It is not a clinical tumor boundary.")
            try:
                heatmap = grad_cam(model, image, device)
                st.image(overlay_heatmap(image, heatmap), caption="Grad-CAM visualization", use_container_width=True)
            except Exception as exc:
                st.warning(f"Grad-CAM could not be generated for this image: {exc}")

            st.warning(
                "Model probability is not clinical certainty. CereScan is a "
                "student research prototype and must not be used for diagnosis."
            )
else:
    st.info("Upload an MRI image to begin.")

st.divider()
st.caption(
    "CereScan is an educational/research prototype. Do not use it to "
    "diagnose, treat, or make clinical decisions."
)
