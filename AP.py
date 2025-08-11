import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import io
import tensorflow as tf
import joblib
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

# Get the current directory (where the script is located)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else os.getcwd()

# Define model paths relative to current directory
MODEL_PATHS = {
    'gan_model': os.path.join(CURRENT_DIR, 'generator_best (2).pth'),
    'cnn_model': os.path.join(CURRENT_DIR, 'best_fire_smoke_cnn.h5'),
    'vae_model_dir': os.path.join(CURRENT_DIR, 'model'),
    'vae_encoder': os.path.join(CURRENT_DIR, 'model', 'thermal_encoder.keras'),
    'vae_decoder': os.path.join(CURRENT_DIR, 'model', 'thermal_decoder.keras'),
    'vae_scaler': os.path.join(CURRENT_DIR, 'model', 'thermal_scaler.pkl'),
    'vae_threshold': os.path.join(CURRENT_DIR, 'model', 'anomaly_threshold.npy'),
}

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

# Function to load the GAN model
@st.cache_resource
def load_generator_model():
    model_path = MODEL_PATHS['gan_model']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    
    try:
        if not os.path.exists(model_path):
            st.error(f"GAN model file not found: {model_path}")
            st.info("Expected file: generator_best (2).pth in the same directory as this script")
            return None, device
            
        # Load the model with better error handling
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if we need to adjust keys for compatibility
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            st.warning(f"Standard loading failed: {str(e)}. Trying alternative loading method...")
            
            # Try removing 'module.' prefix if it exists
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            try:
                model.load_state_dict(new_state_dict)
            except Exception as e2:
                st.error(f"Alternative loading also failed: {str(e2)}")
                return None, device
        
        model.eval()
        st.success("GAN model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"Error loading GAN model: {str(e)}")
        return None, device

# Function to process the image and generate post-fire image
def generate_post_fire_image(model, image, device):
    if model is None:
        st.error("GAN model not loaded properly")
        return None
        
    try:
        # Define the same transforms used during training with better interpolation
        transform = transforms.Compose([
            transforms.Resize((512, 512), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Apply transforms to the input image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate post-fire image
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert output tensor to PIL image
        output_image = output_tensor.squeeze(0).cpu()
        # Denormalize
        output_image = (output_image + 1) / 2
        output_image = transforms.ToPILImage()(output_image)
        
        return output_image
    except Exception as e:
        st.error(f"Error generating post-fire image: {str(e)}")
        return None

# ----------------------------
# CNN Classification Functions
# ----------------------------

# Custom function to get model input shape for debugging
def get_model_input_shape(model):
    try:
        config = model.get_config()
        if 'layers' in config and len(config['layers']) > 0:
            first_layer = config['layers'][0]
            if 'batch_input_shape' in first_layer['config']:
                input_shape = first_layer['config']['batch_input_shape']
                return input_shape[1:]
    except:
        pass
    
    return (224, 224, 3)

# Function to load the CNN model
@st.cache_resource
def load_cnn_model():
    model_path = MODEL_PATHS['cnn_model']
    try:
        if not os.path.exists(model_path):
            st.error(f"CNN model file not found: {model_path}")
            st.info("Expected file: best_fire_smoke_cnn.h5 in the same directory as this script")
            return None
            
        # Try different loading options
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            st.warning(f"Standard loading failed: {str(e1)}. Trying alternative loading method...")
            
            try:
                with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                    model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e2:
                st.error(f"Alternative loading also failed: {str(e2)}")
                return None
        
        input_shape = get_model_input_shape(model)
        st.success(f"CNN model loaded successfully! Expected input shape: {input_shape}")
        
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        return None

# Function to preprocess image for CNN
def preprocess_image_for_cnn(image, model):
    input_shape = None
    
    try:
        config = model.get_config()
        if 'layers' in config and len(config['layers']) > 0:
            first_layer = config['layers'][0]
            if 'batch_input_shape' in first_layer['config']:
                input_shape = first_layer['config']['batch_input_shape'][1:3]
    except:
        pass
    
    if input_shape is None:
        try:
            input_shape = model.input_shape[1:3]
        except:
            input_shape = (224, 224)
    
    st.info(f"Using target size based on model input: {input_shape}")
    
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_array[:, :, :3]
    
    img_resized = cv2.resize(img_rgb, input_shape)
    img_normalized = img_resized / 255.0
    
    st.info(f"Preprocessed image shape: {img_normalized.shape}")
    
    return img_normalized

# Function to classify image using CNN
def classify_image_with_cnn(model, image):
    class_labels = ['fire', 'nofire', 'smoke', 'smokefire']
    
    processed_img = preprocess_image_for_cnn(image, model)
    
    if processed_img is None:
        return {
            'class': 'error',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in class_labels}
        }
    
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
        st.error(f"Error during prediction: {str(e)}")
        st.error("Input shape: " + str(input_array.shape))
        
        return {
            'class': 'error',
            'confidence': 0.0,
            'probabilities': {label: 0.0 for label in class_labels}
        }

# Function to create annotated image with classification results
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

# Function to display classification results
def display_classification_results(results, image, column):
    with column:
        st.subheader("Classification Results")
        
        if results['class'] == 'error':
            st.error("Classification failed. See error message above.")
            return
        
        class_name = results['class']
        confidence = results['confidence'] * 100
        
        class_icons = {
            'fire': '🔥',
            'nofire': '✅',
            'smoke': '💨',
            'smokefire': '🔥💨'
        }
        
        class_colors = {
            'fire': 'red',
            'nofire': 'green',
            'smoke': 'gray',
            'smokefire': 'orange'
        }
        
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
# VAE Anomaly Detection Functions
# ----------------------------

# Custom SamplingLayer for VAE
class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Register the custom layer
tf.keras.utils.get_custom_objects().update({'SamplingLayer': SamplingLayer})

# Function to load VAE models
@st.cache_resource
def load_vae_models():
    model_dir = MODEL_PATHS['vae_model_dir']
    
    try:
        if not os.path.exists(model_dir):
            st.error(f"VAE model directory not found: {model_dir}")
            st.info("Expected directory: 'model' in the same directory as this script")
            return None, None, None, None
        
        # List all files in the directory for debugging
        all_files = os.listdir(model_dir)
        st.success(f"Found {len(all_files)} files in model directory: {all_files}")
        
        # Load scaler
        scaler = None
        scaler_path = MODEL_PATHS['vae_scaler']
        
        if os.path.exists(scaler_path):
            try:
                st.info(f"Loading scaler from {os.path.basename(scaler_path)}")
                scaler = joblib.load(scaler_path)
                st.success("Scaler loaded successfully")
            except Exception as e:
                st.warning(f"Failed to load scaler: {str(e)}")
        
        if scaler is None:
            st.warning("No scaler found. Creating a default StandardScaler.")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        # Load encoder
        encoder = None
        encoder_path = MODEL_PATHS['vae_encoder']
        
        if os.path.exists(encoder_path):
            st.info(f"Loading encoder from {os.path.basename(encoder_path)}")
            
            try:
                encoder = tf.keras.models.load_model(
                    encoder_path, 
                    custom_objects={'SamplingLayer': SamplingLayer}
                )
                st.success("Encoder loaded successfully")
            except Exception as e1:
                st.warning(f"Standard loading failed: {str(e1)}")
                
                try:
                    with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                        encoder = tf.keras.models.load_model(encoder_path)
                    st.success("Encoder loaded with custom object scope")
                except Exception as e2:
                    st.error(f"Alternative loading also failed: {str(e2)}")
        
        if encoder is None:
            st.error("Could not load encoder model. VAE functionality will not work.")
            return None, None, None, None
        
        # Load decoder
        decoder = None
        decoder_path = MODEL_PATHS['vae_decoder']
        
        if os.path.exists(decoder_path):
            st.info(f"Loading decoder from {os.path.basename(decoder_path)}")
            
            try:
                decoder = tf.keras.models.load_model(decoder_path)
                st.success("Decoder loaded successfully")
            except Exception as e1:
                st.warning(f"Standard loading failed: {str(e1)}")
                
                try:
                    with tf.keras.utils.custom_object_scope({'SamplingLayer': SamplingLayer}):
                        decoder = tf.keras.models.load_model(decoder_path)
                    st.success("Decoder loaded with custom object scope")
                except Exception as e2:
                    st.error(f"Alternative loading also failed: {str(e2)}")
        
        if decoder is None:
            st.error("Could not load decoder model. VAE functionality will not work.")
            return None, None, None, None
        
        # Load threshold
        threshold = None
        threshold_path = MODEL_PATHS['vae_threshold']
        
        if os.path.exists(threshold_path):
            try:
                threshold = float(np.load(threshold_path))
                st.success(f"Threshold loaded: {threshold}")
            except Exception as e:
                st.warning(f"Failed to load threshold: {str(e)}")
        
        if threshold is None:
            st.warning("No threshold file found. Using default value of 0.1")
            threshold = 0.1
        
        return scaler, encoder, decoder, threshold
    
    except Exception as e:
        st.error(f"Error in load_vae_models: {str(e)}")
        return None, None, None, None

# Function to process CSV data and detect anomalies
def predict_anomaly_from_csv(df, models):
    scaler, encoder, decoder, threshold = models
    
    if encoder is None or decoder is None:
        st.error("VAE models not properly loaded. Cannot perform anomaly detection.")
        return None
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("No numeric columns found in the CSV file. VAE requires numeric data.")
            return None
        
        st.info(f"Using {len(numeric_df.columns)} numeric columns for anomaly detection: {', '.join(numeric_df.columns)}")
        
        features = numeric_df.values
        
        try:
            scaled_input = scaler.transform(features)
        except Exception as e:
            st.error(f"Error scaling input data: {str(e)}")
            st.warning("Fitting a new scaler to the data. This may affect anomaly detection accuracy.")
            
            from sklearn.preprocessing import StandardScaler
            new_scaler = StandardScaler()
            scaled_input = new_scaler.fit_transform(features)
        
        try:
            encoder_input_shape = encoder.input_shape
            st.info(f"Encoder expects input shape: {encoder_input_shape}")
            st.info(f"Provided data shape: {scaled_input.shape}")
            
            encoder_outputs = encoder.predict(scaled_input, verbose=0)
            
            if isinstance(encoder_outputs, list) and len(encoder_outputs) >= 3:
                z_mean_input, z_log_var_input, z_input = encoder_outputs
            elif isinstance(encoder_outputs, list) and len(encoder_outputs) == 2:
                z_mean_input, z_log_var_input = encoder_outputs
                batch = z_mean_input.shape[0]
                dim = z_mean_input.shape[1]
                epsilon = np.random.normal(size=(batch, dim))
                z_input = z_mean_input + np.exp(0.5 * z_log_var_input) * epsilon
            else:
                z_input = encoder_outputs
                z_mean_input = encoder_outputs
                z_log_var_input = np.zeros_like(z_mean_input)
        except Exception as e:
            st.error(f"Error in encoder prediction: {str(e)}")
            return None
        
        try:
            decoder_input_shape = decoder.input_shape
            st.info(f"Decoder expects input shape: {decoder_input_shape}")
            st.info(f"Latent representation shape: {z_input.shape}")
            
            reconstructed_input = decoder.predict(z_input, verbose=0)
        except Exception as e:
            st.error(f"Error in decoder prediction: {str(e)}")
            return None
        
        mse_input = np.mean(np.square(scaled_input - reconstructed_input), axis=1)
        
        is_anomaly = mse_input > threshold
        
        anomaly_scores = mse_input / threshold
        
        return {
            'reconstruction_errors': mse_input,
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'reconstructed_data': reconstructed_input,
            'latent_representations': z_input,
            'z_mean': z_mean_input,
            'z_log_var': z_log_var_input
        }
    except Exception as e:
        st.error(f"Error in anomaly prediction: {str(e)}")
        return None

# Function to display CSV anomaly detection results
def display_csv_anomaly_results(results, df):
    pd.set_option("styler.render.max_elements", 1500000)
    
    anomaly_count = np.sum(results['is_anomaly'])
    total_count = len(results['is_anomaly'])
    anomaly_percentage = (anomaly_count / total_count) * 100
    
    st.subheader("Anomaly Detection Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", total_count)
    col2.metric("Anomalies Detected", anomaly_count)
    col3.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
    
    fig_gauge = plt.figure(figsize=(6, 3))
    ax = fig_gauge.add_subplot(111)
    gauge_colors = ['green', 'yellow', 'orange', 'red']
    thresholds = [0, 10, 30, 60, 100]
    gauge_color = gauge_colors[0]
    for i in range(1, len(thresholds)):
        if anomaly_percentage >= thresholds[i-1] and anomaly_percentage <= thresholds[i]:
            gauge_color = gauge_colors[i-1]
    ax.barh(1, 100, color='lightgray', height=0.5)
    ax.barh(1, anomaly_percentage, color=gauge_color, height=0.5)
    ax.text(50, 1, f"{anomaly_percentage:.2f}% Anomalies", ha='center', va='center', fontweight='bold')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 100)
    ax.set_frame_on(False)
    st.pyplot(fig_gauge)
    
    result_df = df.copy()
    result_df['Reconstruction_Error'] = results['reconstruction_errors']
    result_df['Anomaly_Score'] = results['anomaly_scores']
    result_df['Is_Anomaly'] = results['is_anomaly']
    
    if len(result_df) > 10000:
        st.warning(f"Dataset is large ({len(result_df)} rows). Showing only anomalies and a sample of normal records.")
        
        anomalies_df = result_df[result_df['Is_Anomaly']]
        non_anomalies = result_df[~result_df['Is_Anomaly']].sample(min(1000, len(result_df[~result_df['Is_Anomaly']])))
        display_df = pd.concat([anomalies_df, non_anomalies]).sort_index()
        
        def highlight_anomalies(s):
            is_anomaly = s['Is_Anomaly']
            return ['background-color: rgba(255, 0, 0, 0.2)' if is_anomaly else '' for _ in s]
        
        styled_df = display_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df)
    else:
        def highlight_anomalies(s):
            is_anomaly = s['Is_Anomaly']
            return ['background-color: rgba(255, 0, 0, 0.2)' if is_anomaly else '' for _ in s]
        
        styled_df = result_df.style.apply(highlight_anomalies, axis=1)
        st.dataframe(styled_df)
    
    st.subheader("Reconstruction Loss Distribution")
    fig_loss = plt.figure(figsize=(10, 6))
    plt.hist(results['reconstruction_errors'], bins=50, alpha=0.7, color='blue', label='All samples')
    
    if anomaly_count > 0:
        plt.hist(results['reconstruction_errors'][results['is_anomaly']], 
                 bins=20, alpha=0.7, color='red', label='Anomalies')
        
    plt.axvline(x=results['threshold'], color='red', linestyle='--', 
                label=f'Threshold: {results["threshold"]:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(fig_loss)
    
    if results['latent_representations'].shape[1] >= 2:
        st.subheader("Latent Space Representation")
        fig_latent = plt.figure(figsize=(10, 8))
        
        normal_mask = ~results['is_anomaly']
        anomaly_mask = results['is_anomaly']
        
        plt.scatter(
            results['latent_representations'][normal_mask, 0],
            results['latent_representations'][normal_mask, 1],
            c='blue', alpha=0.5, label='Normal'
        )
        
        if np.any(anomaly_mask):
            plt.scatter(
                results['latent_representations'][anomaly_mask, 0],
                results['latent_representations'][anomaly_mask, 1],
                c='red', alpha=0.7, label='Anomaly'
            )
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('2D Projection of Latent Space')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_latent)
        
        # Add download buttons for results
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="Download Results as CSV",
            data=csv_buffer,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )

# ----------------------------
# Main App Interface
# ----------------------------
def main():
    st.title("🔥 Wildfire Analysis Tool")
    st.write("An all-in-one tool for wildfire detection, simulation, and analysis")
    
    # Display current directory and expected files
    with st.expander("Model File Status"):
        st.write(f"**Current Directory:** {CURRENT_DIR}")
        st.write("**Expected Model Files:**")
        
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                st.success(f"✅ {model_name}: {os.path.basename(model_path)} - Found")
            else:
                st.error(f"❌ {model_name}: {os.path.basename(model_path)} - Missing")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["GAN Fire Simulation", "CNN Classification", "VAE Anomaly Detection"])
    
    with tab1:
        st.header("GAN Fire Simulation")
        st.write("Generate post-fire scenarios from pre-fire images")
        
        # Load GAN model
        with st.spinner("Loading GAN model..."):
            gan_model, device = load_generator_model()
        
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
                        try:
                            post_fire_image = generate_post_fire_image(gan_model, image, device)
                            
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
                        except Exception as e:
                            st.error(f"Error generating post-fire image: {str(e)}")
        else:
            st.warning("GAN model not loaded. Please ensure 'generator_best (2).pth' is in the same directory as this script.")
    
    with tab2:
        st.header("CNN Classification")
        st.write("Classify wildfire images into fire, smoke, both, or none")
        
        # Load CNN model
        with st.spinner("Loading CNN model..."):
            cnn_model = load_cnn_model()
        
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
                            st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("CNN model not loaded. Please ensure 'best_fire_smoke_cnn.h5' is in the same directory as this script.")
    
    with tab3:
        st.header("VAE Anomaly Detection")
        
        st.info("""
        This module uses a Variational Autoencoder (VAE) to detect anomalies in your CSV data.
        Upload a CSV file containing numeric features, and the model will identify potential anomalies.
        
        **How it works:** The VAE learns the normal pattern of your data and flags samples that deviate significantly.
        """)
        
        # Load VAE models
        with st.spinner("Loading VAE models..."):
            vae_models = load_vae_models()
        
        if all(model is not None for model in vae_models):
            uploaded_csv = st.file_uploader("Upload CSV file for anomaly detection", type=['csv'])
            
            if uploaded_csv is not None:
                try:
                    df = pd.read_csv(uploaded_csv)
                    
                    with st.expander("CSV Data Preview"):
                        st.dataframe(df.head())
                    
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_columns:
                        st.error("No numeric columns found in the CSV file. VAE requires numeric data.")
                    else:
                        if st.button("Detect Anomalies"):
                            with st.spinner("Detecting anomalies..."):
                                try:
                                    anomaly_results = predict_anomaly_from_csv(df, vae_models)
                                    
                                    if anomaly_results is not None:
                                        display_csv_anomaly_results(anomaly_results, df)
                                except Exception as e:
                                    st.error(f"Error during anomaly detection: {str(e)}")
                                    st.info("Try uploading a different CSV or check if the data format matches what the model expects.")
                except Exception as e:
                    st.error(f"Error processing CSV file: {str(e)}")
        else:
            st.warning("VAE models not loaded properly. Please ensure the 'model' directory exists with all required files:")
            st.write("- thermal_encoder.keras")
            st.write("- thermal_decoder.keras") 
            st.write("- thermal_scaler.pkl")
            st.write("- anomaly_threshold.npy")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2023 Wildfire Analysis Tool | Created for research purposes")

if __name__ == "__main__":
    main()
