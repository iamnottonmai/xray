import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import gdown
import os
import importlib.util

# ============================
# CONFIG
# ============================
MODEL_URL = "https://drive.google.com/uc?id=1zTS45HMzZvaEEcFycW61LE6gnSbC692J"
MODEL_PATH = "srcnn_epoch99.pth"
SRCNN_PATH = "SRCNN.py"
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
st.set_page_config(page_title="SRCNN Super-Resolution")
st.title("🔍 SRCNN Image Super-Resolution")
st.write("Upload a **grayscale** image. The model expects grayscale input.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load original image in grayscale mode
    original_img = Image.open(uploaded_file).convert("L")
    
    # Transform: resize to 256x256 & to tensor for model input
    transform_infer = T.Compose([
        T.Resize((256, 256), interpolation=Image.BICUBIC),
        T.ToTensor(),
    ])
    input_tensor = transform_infer(original_img).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.clamp(output_tensor, 0, 1)

    # Prepare images for display
    # Original image resized to 1024x1024 for display
    original_vis = original_img.resize((1024, 1024), Image.BICUBIC)

    # Output tensor to PIL, then resize for display
    output_img = T.ToPILImage()(output_tensor.squeeze().cpu())
    output_vis = output_img.resize((1024, 1024), Image.BICUBIC)

    # Display images side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_vis, caption="Original Grayscale Image (Resized 1024x1024)", use_column_width=True)
    with col2:
        st.image(output_vis, caption="SRCNN Super-Resolved Output", use_column_width=True)

    st.write("Inference done successfully.")
