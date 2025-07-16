import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gdown
import os

# ============================
# CONFIG
# ============================
MODEL_URL = "https://drive.google.com/uc?id=1zTS45HMzZvaEEcFycW61LE6gnSbC692J"
MODEL_PATH = "srcnn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# DOWNLOAD MODEL IF NEEDED
# ============================
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ============================
# SRCNN Model
# ============================
from srcnn import SRCNN  # <--- changed line
import importlib.util

spec = importlib.util.spec_from_file_location("srcnn", "srcnn.py")
srcnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srcnn_module)
SRCNN = srcnn_module.SRCNN

model = SRCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
# ============================
# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="🧠 SRCNN Image Super-Resolution")
st.title("🔍 SRCNN Super-Resolution Demo")
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

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Low-Res Input", use_column_width=True)
    with col2:
        st.image(output, caption="SRCNN Output", use_column_width=True, clamp=True)

