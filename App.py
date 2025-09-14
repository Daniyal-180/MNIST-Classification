import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Model (simple CNN for 28x28 grayscale digits 0–9)
# -------------------------
class NumberCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # EXTRA LAYER
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)   # smaller spatial size -> 1152
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))     # conv3 included
        x = self.dropout(x)
        x = x.view(-1, 128 * 3 * 3)              # matches 1152
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------
# Load weights helper
# -------------------------
@st.cache_resource
def load_model(weights_path: str):
    model = NumberCNN().to(device)
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Preprocessing
# -------------------------
def preprocess_image(pil_img: Image.Image,
                     invert: bool = False,
                     auto_contrast: bool = True,
                     target_size=(28, 28)) -> torch.Tensor:
    """
    Returns a 4D tensor [1,1,28,28] normalized to mean=0.5, std=0.5
    """
    # Convert to grayscale
    img = pil_img.convert("L")

    # Optional auto-contrast (helps photos)
    if auto_contrast:
        img = ImageOps.autocontrast(img)

    # Optional invert (MNIST is white-on-black; your paper/photo may be black-on-white)
    if invert:
        img = ImageOps.invert(img)

    # Resize to 28x28 (keep it simple; center-crop/pad can be added if needed)
    img = img.resize(target_size, Image.BILINEAR)

    # To tensor and normalize
    to_tensor = transforms.ToTensor()  # scales to [0,1], shape [1,28,28]
    tensor = to_tensor(img)
    normalize = transforms.Normalize((0.5,), (0.5,))  # (x-0.5)/0.5 -> [-1,1]
    tensor = normalize(tensor).unsqueeze(0)  # [1,1,28,28]
    return tensor

# -------------------------
# Prediction
# -------------------------
def predict_digit(model: nn.Module, tensor_4d: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor_4d.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Digit Classifier (PyTorch + Streamlit)", page_icon="🔢", layout="centered")

st.title("🔢 Number Classification — PyTorch + Streamlit")
st.caption("Upload an image or capture from your webcam to classify digits (0–9).")

with st.sidebar:
    st.subheader("Settings")
    weights_path = st.text_input("Model weights path", value="digit_classifier.pth")
    invert_opt = st.checkbox("Invert colors (black ↔ white)", value=True)
    autocontrast_opt = st.checkbox("Auto-contrast", value=True)
    st.write(f"Device: **{device}**")

model, load_err = load_model(weights_path)

if load_err:
    st.error(f"Could not load model from '{weights_path}'.\n\nDetails: {load_err}")
    st.info("Tip: ensure your trained weights file matches the model architecture here "
            "(28×28 grayscale digits, mean=0.5, std=0.5).")
else:
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam"])

    def handle_image(pil_img):
        st.subheader("Input Preview")
        st.image(pil_img, caption="Original", use_container_width=True)

        tensor = preprocess_image(pil_img, invert=invert_opt, auto_contrast=autocontrast_opt)
        pred, probs = predict_digit(model, tensor)

        # Show processed 28x28 thumbnail
        proc_vis = tensor.clone().cpu()
        # unnormalize for display: x*0.5+0.5 -> [0,1]
        proc_vis = proc_vis * 0.5 + 0.5
        proc_img = transforms.ToPILImage()(proc_vis.squeeze(0))
        st.image(proc_img.resize((140, 140), Image.NEAREST), caption="Processed (28×28)", width=140)

        st.subheader("Prediction")
        st.metric("Predicted Digit", str(pred))

        st.subheader("Class Probabilities")
        st.bar_chart(
            {f"{i}": float(probs[i]) for i in range(10)},
            height=240
        )

    with tab1:
        file = st.file_uploader("Drag & drop an image of a single digit (jpg/png)", type=["png", "jpg", "jpeg"])
        if file is not None:
            try:
                img = Image.open(io.BytesIO(file.read()))
                handle_image(img)
            except Exception as e:
                st.error(f"Failed to read image: {e}")

    with tab2:
        shot = st.camera_input("Take a photo of a digit")
        if shot is not None:
            try:
                img = Image.open(shot)
                handle_image(img)
            except Exception as e:
                st.error(f"Failed to process webcam image: {e}")

    st.divider()
    st.markdown(
        """
        **Notes**
        - Your model should be trained on **28×28 grayscale** digits with normalization **(mean=0.5, std=0.5)**.
        - If predictions look inverted (e.g., background confused with digit), toggle **Invert colors**.
        - For better real-world performance, train a CNN (as above) on MNIST and include mild data augmentation.
        """
    )
