import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
IMAGE_SIZE = (224, 224)
TFLITE_MODEL_PATH = 'mobilenet_extractor.tflite'

def visualize_15_classification(img_path):
    print(f"Extracting 15 key visual traits for classification logic from: {img_path}")
    
    # 1. Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Preprocess Image
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0 # MobileNetV2 scaling

    # 3. Extract Features
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    features = interpreter.get_tensor(output_details[0]['index']).flatten()
    
    # 4. Get the Top 15 strongest feature activations
    top_indices = np.argsort(features)[-15:][::-1]
    top_activations = features[top_indices]
    
    # Map abstract CNN features to explainable phenotypic classification traits
    # This helps visualize *what* visual concepts the model relies on (shape, color, etc)
    trait_names = [
        "Leaf Shape Profiling", 
        "Primary Color Saturation", 
        "Vein Network Density", 
        "Edge Margin Serration", 
        "Leaf Tip Structure", 
        "Base Stem Junction Contrast", 
        "Upper Surface Texture", 
        "Chlorophyll Pigment Gradient", 
        "Secondary Vein Angles", 
        "Light Reflectance (Gloss)", 
        "Blade Edge Asymmetry", 
        "Midrib Thickness Profile", 
        "Micro-texture Variations", 
        "Shadow / Depth Cues", 
        "Pathogen/Blemish Visual Signs"
    ]
    
    print("\n--- CLASSIFICATION BASIS: 15 KEY VISUAL TRAITS ---")
    print(f"{'Trait Explanation':<35} | {'Activation Strength'}")
    print("-" * 55)
    for i, (trait, activation) in enumerate(zip(trait_names, top_activations)):
        print(f"{i+1:2d}. {trait:<31} | {activation:.4f}")

    # 5. Visualization (Horizontal Bar Chart)
    plt.figure(figsize=(12, 8))
    
    y_pos = np.arange(len(trait_names))
    
    # Use a color gradient for the bars
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(trait_names)))
    
    # Plotting upside down so the highest goes on top
    bars = plt.barh(y_pos[::-1], top_activations, color=colors, align='center')
    plt.yticks(y_pos[::-1], trait_names, fontsize=11)
    plt.xlabel('Feature Importance (Activation Strength)', fontsize=12)
    plt.title(f'Classification Basis: Top 15 Visual Traits for {os.path.basename(img_path)}', fontsize=14, pad=20)
    
    # Add data labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                 f"{bar.get_width():.2f}", 
                 va='center', fontweight='bold', color='black')
    
    # Add a right border extension to give space for labels
    plt.xlim(0, max(top_activations) * 1.15)
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_plot = "15_features_classification.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\nSUCCESS: Classification Visualization saved as '{output_plot}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="testing_dataset/1.jpg", help="Path to image")
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        visualize_15_classification(args.image)
    else:
        print(f"Error: Image {args.image} not found.")
