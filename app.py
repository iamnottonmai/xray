import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown
import os
import importlib.util

# ============================
# CONFIG
# ============================
MODEL_URL = "https://drive.google.com/uc?id=1zTS45HMzZvaEEcFycW61LE6gnSbC692J"
MODEL_PATH = "srcnn_epoch50.pth"
SRCNN_PATH = "SRCNN.py"  # Make sure this matches your file name exactly
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# DOWNLOAD MODEL IF NEEDED
# ============================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ============================
# LOAD SRCNN CLASS FROM SRCNN.py
# ============================
if os.path.exists(SRCNN_PATH):
    spec = importlib.util.spec_from_file_location("SRCNN", SRCNN_PATH)
    srcnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srcnn_module)
    SRCNN = srcnn_module.SRCNN
else:
    st.error(f"Could not find {SRCNN_PATH}. Make sure it exists in the same directory as app.py.")
    st.stop()

# ============================
# LOAD MODEL
# ============================
model = SRCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="  SRCNN Image Super-Resolution")
st.title("  SRCNN Super-Resolution Demo")
st.write("Upload a **grayscale** image and see the enhanced version.")

uploaded_file = st.file_uploader("Upload a low-res image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load and preprocess
    image = Image.open(uploaded_file).convert("L")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.clamp(output, 0, 1).squeeze().cpu().numpy()

    # Prepare images for display
    input_display = image.resize((64, 64))
    output_img = Image.fromarray((output * 255).astype(np.uint8))
    output_img = output_img.resize((64, 64))

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(input_display, caption="Low-Res Input (64×64 px)", use_container_width=False)
    with col2:
        st.image(output_img, caption="SRCNN Output (64×64 px)", use_container_width=False)
