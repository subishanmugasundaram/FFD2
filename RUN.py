import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import io
import tensorflow as tf
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import cv2

# Set page configuration
st.set_page_config(
    page_title="Wildfire Analysis Tool",
    page_icon="🔥",
    layout="wide",
)

# Add custom styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #ff7043;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff5722;
    }
    .reportview-container {
        background-color: #f0f2f6;
    }
    h1 {
        color: #ff5722;
    }
    h2 {
        color: #ff7043;
    }
    h3 {
        color: #ff8a65;
    }
</style>
""", unsafe_allow_html=True)

# Universal model directory path
MODEL_DIR = "."

# ----------------------------
# U-Net Generator for GAN
# ----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.enc1 = self._encoder_block(in_channels, 64, use_batchnorm=False)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        self.enc5 = self._encoder_block(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.dec5 = self._decoder_block(1024, 512)
        self.dec4 = self._decoder_block(1024, 256)
        self.dec3 = self._decoder_block(512, 128) 
        self.dec2 = self._decoder_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _encoder_block(self, in_channels, out_channels, use_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = self.dec5(torch.cat([b, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        output = self.final(torch.cat([d2, e1], dim=1))
        return output

@st.cache_resource
def load_generator_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    
    try:
        if not os.path.exists(model_path):
            st.error(f"GAN model not found: {model_path}")
            return None, device
            
        state_dict = torch.load(model_path, map_location=device)
        
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"Standard loading failed, trying alternative...")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        
        model.eval()
        st.success("✅ GAN model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading GAN model: {str(e)}")
        return None, device

def generate_post_fire_image(model, image, device):
    if model is None:
        st.error("❌ GAN model not loaded")
        return None
        
    try:
        transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        output_image = output_tensor.squeeze(0).cpu()
        output_image = (output_image + 1) / 2
        output_image = transforms.ToPILImage()(output_image)
        
        return output_image
    except Exception as e:
        st.error(f"❌ Error generating image: {str(e)}")
        return None

# ----------------------------
# CNN Classification Functions
# ----------------------------

@st.cache_resource
def load_cnn_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"CNN model not found: {model_path}")
            return None
            
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            st.warning(f"Standard loading failed: {str(e1)}")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                st.error(f"Alternative loading failed: {str(e2)}")
                return None
        
        st.success("✅ CNN model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading CNN model: {str(e)}")
        return None

def preprocess_image_for_cnn(image, model):
    try:
        input_shape = model.input_shape[1:3]
    except:
        input_shape = (224, 224)
    
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array[:, :, :3]
    
    img_resized = cv2.resize(img_rgb, input_shape)
    img_normalized = img_resized / 255.0
    
    return img_normalized

def classify_image_with_cnn(model, image):
    class_labels = ['fire', 'nofire', 'smoke', 'smokefire']
    
    processed_img = preprocess_image_for_cnn(image, model)
    if processed_img is None:
        return {'class': 'error', 'confidence': 0.0, 'probabilities': {label: 0.0 for label in class_labels}}
    
    input_array = np.expand_dims(processed_img, axis=0)
    
    try:
        prediction = model.predict(input_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        }
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return {'class': 'error', 'confidence': 0.0, 'probabilities': {label: 0.0 for label in class_labels}}

def create_annotated_image(image, class_name, confidence):
    img_array = np.array(image)
    annotated_img = img_array.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{class_name} ({confidence:.1f}%)"
    text_x = 10
    text_y = annotated_img.shape[0] - 20
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
    cv2.rectangle(annotated_img, 
                  (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + 5), 
                  (0, 0, 0), -1)
    
    cv2.putText(annotated_img, text, (text_x, text_y), 
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return Image.fromarray(annotated_img)

def display_classification_results(results, image, column):
    with column:
        st.subheader("Classification Results")
        
        if results['class'] == 'error':
            st.error("❌ Classification failed")
            return
        
        class_name = results['class']
        confidence = results['confidence'] * 100
        
        class_icons = {'fire': '🔥', 'nofire': '✅', 'smoke': '💨', 'smokefire': '🔥💨'}
        class_colors = {'fire': 'red', 'nofire': 'green', 'smoke': 'gray', 'smokefire': 'orange'}
        
        icon = class_icons.get(class_name, '❓')
        color = class_colors.get(class_name, 'blue')
        
        st.markdown(
            f"<h3 style='color: {color};'>{icon} {class_name.upper()} ({confidence:.2f}%)</h3>",
            unsafe_allow_html=True
        )
        
        st.write("### Class Probabilities")
        probs = results['probabilities']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(probs.keys())
        values = list(probs.values())
        colors = [class_colors.get(cls, 'blue') for cls in classes]
        bars = ax.bar(classes, [v * 100 for v in values], color=colors)
        
        ax.set_ylabel('Probability (%)')
        ax.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        annotated_img = create_annotated_image(image, class_name, confidence)
        st.subheader("Annotated Image")
        st.image(annotated_img, use_column_width=True)
        
        buf = BytesIO()
        annotated_img.save(buf, format="PNG")
        buf.seek(0)
        
        st.download_button(
            label="Download Annotated Image",
            data=buf,
            file_name=f"classified_{class_name}.png",
            mime="image/png"
        )

# ----------------------------
# Main App Interface
# ----------------------------
def main():
    st.title("🔥 Wildfire Analysis Tool")
    st.write("An all-in-one tool for wildfire detection and simulation")
    
    tab1, tab2 = st.tabs(["GAN Fire Simulation", "CNN Classification"])
    
    with tab1:
        st.header("GAN Fire Simulation")
        st.write("Generate post-fire scenarios from pre-fire images")
        
        gan_model_files = ["generator_best (2).pth", "generator_best.pth", "generator.pth"]
        gan_model_path = None
        
        for gan_file in gan_model_files:
            potential_path = os.path.join(MODEL_DIR, gan_file)
            if os.path.exists(potential_path):
                gan_model_path = potential_path
                break
        
        if gan_model_path is None:
            gan_model_path = os.path.join(MODEL_DIR, "generator_best (2).pth")
        
        gan_model_path = st.text_input("GAN Model Path", value=gan_model_path)
        
        with st.spinner("Loading GAN model..."):
            gan_model, device = load_generator_model(gan_model_path)
        
        if gan_model is not None:
            uploaded_file = st.file_uploader("Upload a pre-fire image", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    display_img = image.resize((512, 512), Image.LANCZOS)
                    st.image(display_img, width=512)
                
                if st.button("Generate Post-Fire Scenario"):
                    with st.spinner("Generating..."):
                        post_fire_image = generate_post_fire_image(gan_model, image, device)
                        
                        if post_fire_image:
                            with col2:
                                st.subheader("Generated Post-Fire Scenario")
                                post_fire_display = post_fire_image.resize((512, 512), Image.LANCZOS)
                                st.image(post_fire_display, width=512)
                                
                                buf = BytesIO()
                                post_fire_image.save(buf, format="PNG")
                                buf.seek(0)
                                
                                st.download_button(
                                    label="Download Generated Image",
                                    data=buf,
                                    file_name="post_fire_scenario.png",
                                    mime="image/png"
                                )
    
    with tab2:
        st.header("CNN Classification")
        st.write("Classify wildfire images into fire, smoke, both, or none")
        
        cnn_model_files = ["best_fire_smoke_cnn.h5", "best_fire_smoke_cnn", "fire_smoke_cnn.h5"]
        cnn_model_path = None
        
        for cnn_file in cnn_model_files:
            potential_path = os.path.join(MODEL_DIR, cnn_file)
            if os.path.exists(potential_path):
                cnn_model_path = potential_path
                break
        
        if cnn_model_path is None:
            cnn_model_path = os.path.join(MODEL_DIR, "best_fire_smoke_cnn.h5")
        
        cnn_model_path = st.text_input("CNN Model Path", value=cnn_model_path)
        
        with st.spinner("Loading CNN model..."):
            cnn_model = load_cnn_model(cnn_model_path)
        
        if cnn_model is not None:
            uploaded_file = st.file_uploader("Upload an image for classification", type=['jpg', 'jpeg', 'png'], key="cnn_uploader")
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Image to Classify")
                    st.image(image, use_column_width=True)
                
                if st.button("Classify Image"):
                    with st.spinner("Classifying..."):
                        try:
                            classification_results = classify_image_with_cnn(cnn_model, image)
                            display_classification_results(classification_results, image, col2)
                        except Exception as e:
                            st.error(f"❌ Classification error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Wildfire Analysis Tool | Created for research purposes")

if __name__ == "__main__":
    main()
