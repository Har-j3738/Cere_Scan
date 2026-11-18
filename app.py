import streamlit as st
from PIL import Image

from pipeline import demo_predict, preprocess_image


st.set_page_config(
    page_title="CereScan",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CereScan")
st.subheader("AI-assisted brain MRI analysis")
st.caption("Prototype • Educational / Research Use")

st.sidebar.title("CereScan")
st.sidebar.write("Pipeline prototype v0.1")
st.sidebar.warning(
    "This prototype is not a medical diagnostic system."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file)

    left, right = st.columns(2)

    with left:
        st.subheader("MRI Scan")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Analysis")

        if st.button("🔬 Analyze MRI", type="primary"):
            with st.spinner("Running preprocessing and inference..."):
                processed = preprocess_image(image)
                predicted_class, confidence, predictions = demo_predict(
                    processed
                )

            st.success("Pipeline completed")

            st.metric("Predicted Class", predicted_class)
            st.metric("Demo Confidence", f"{confidence * 100:.1f}%")

            st.subheader("Class Probabilities")

            for label, probability in predictions.items():
                st.progress(
                    probability,
                    text=f"{label}: {probability * 100:.1f}%",
                )

            st.info(
                "These probabilities are placeholders for UI development. "
                "They are not produced by a trained medical AI model."
            )
else:
    st.info("Upload an MRI image to begin.")

st.divider()
st.caption(
    "CereScan is an educational/research prototype and must not be "
    "used as a substitute for professional medical diagnosis."
)
