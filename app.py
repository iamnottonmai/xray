import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import gdown
import os

# ============================
# CONFIGURATION
# ============================
MODEL_URL = "https://drive.google.com/uc?id=1xYLFfrS7VaQr6MEgwN0vA-bSJGL1qf3C"
MODEL_PATH = "srcnn_epoch99.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_IMAGES = {
    "CT Sample 1": "sample_images/IMG_0807.png",
    "CT Sample 2": "sample_images/IMG_0808.png",
    "CT Sample 3": "sample_images/IMG_0809.png",
    "CT Sample 4": "sample_images/IMG_0810.png"
}

# ============================
# SRCNN Model Definition
# ============================
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# ============================
# LOAD MODEL FROM DRIVE (cached)
# ============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = SRCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Load model ONCE when app runs
model = load_model()

# ============================
# PREPROCESS IMAGE
# ============================
def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# ============================
# POSTPROCESS TO PIL
# ============================
def postprocess(tensor: torch.Tensor) -> Image.Image:
    tensor = torch.clamp(tensor.squeeze().cpu(), 0, 1)
    return transforms.ToPILImage()(tensor)

# ============================
# STREAMLIT APP
# ============================
st.title("CT Chest Super-Resolution")
st.write("Upload a **Chest CT image**, or select a sample image to enhance its resolution.")

# --- Sample Images Section ---
with st.expander("Or choose a sample image"):
    selected_sample = st.radio("Select sample image", list(SAMPLE_IMAGES.keys()))

# --- File Upload ---
uploaded_file = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

# --- Load image (sample or upload) ---
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
elif selected_sample:
    image = Image.open(SAMPLE_IMAGES[selected_sample]).convert("L")

# --- Perform inference ---
if image:
    st.image(image.resize((512, 512), resample=Image.BICUBIC), caption="Low-Resolution Input", use_column_width=True)

    input_tensor = preprocess(image)

    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_image = postprocess(output_tensor)

    st.image(output_image.resize((512, 512), resample=Image.BICUBIC), caption="Super-Resolved Output", use_column_width=True)
