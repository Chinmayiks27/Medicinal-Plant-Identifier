import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = (224, 224)
TFLITE_MODEL_PATH = 'mobilenet_extractor.tflite'
PSO_MASK_PATH = 'pso_feature_mask.npy'

def visualize_pipeline(img_path):
    print(f"Generating optimized visual pipeline with clear vertical callouts for: {img_path}")
    
    # 1. Original Image Load
    img_orig = Image.open(img_path).convert('RGB')
    
    # 2. Preprocess
    img_resized = img_orig.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_scaled = (img_array_expanded / 127.5) - 1.0 # -1 to 1 MobileNet scale

    # 3. Model Inference (Get Raw 1280 Features)
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array_scaled)
    interpreter.invoke()
    raw_features = interpreter.get_tensor(output_details[0]['index']).flatten()

    # --- PLOTTING THE PROCESS FLOW ---
    # Increased height even more to ensure text has absolute clearance
    fig = plt.figure(figsize=(18, 14))  
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle('End-to-End Image Processing & Feature Classification Pipeline', fontsize=22, fontweight='bold', y=0.96)
    
    # Step 1: Input Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img_orig)
    ax1.set_title('1. Image Capture', fontsize=16, fontweight='bold', pad=10)
    ax1.text(0.5, -0.15, "(Raw Pixels Captured via App)", ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.axis('off')
    
    # Step 2: Normalization
    ax2 = plt.subplot(2, 3, 2)
    vis_scaled = ((img_array_scaled[0] + 1.0) / 2.0)
    ax2.imshow(vis_scaled)
    ax2.set_title('2. Spatial & Color Normalization', fontsize=16, fontweight='bold', pad=10)
    ax2.text(0.5, -0.15, f"(Rescaled to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3 Tensor)", ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.axis('off')

    # Step 3: Visual CNN Architecture
    ax3 = plt.subplot(2, 3, 3)
    ax3.add_patch(patches.Rectangle((0.1, 0.2), 0.15, 0.6, facecolor='#add8e6', edgecolor='#00008b', lw=2))
    ax3.add_patch(patches.Rectangle((0.35, 0.3), 0.15, 0.4, facecolor='#4169e1', edgecolor='#00008b', lw=2))
    ax3.add_patch(patches.Rectangle((0.6, 0.4), 0.15, 0.2, facecolor='#00008b', edgecolor='#000040', lw=2))
    
    ax3.annotate('', xy=(0.33, 0.5), xytext=(0.27, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
    ax3.annotate('', xy=(0.58, 0.5), xytext=(0.52, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
    ax3.annotate('', xy=(0.88, 0.5), xytext=(0.78, 0.5), arrowprops=dict(arrowstyle="->", lw=3.5, color='#333333'))

    ax3.set_title("3. CNN Feature Extraction", fontsize=16, fontweight='bold', pad=10)
    ax3.text(0.5, 0.85, "MobileNetV2 Layers", ha='center', va='center', fontsize=14, fontweight='bold', color='#00008b')
    ax3.text(0.5, -0.1, "(Dimensionality Reduction Flow)", ha='center', va='center', fontsize=12, color='gray')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Step 4: Output 1280 Features
    ax4 = plt.subplot(2, 3, 4)
    x_val = np.arange(len(raw_features))
    ax4.fill_between(x_val, raw_features, color='#6ca6cd', alpha=0.7)
    ax4.plot(x_val, raw_features, color='#104e8b', lw=0.6)
    
    ax4.set_title('4. Raw Extracted Signal', fontsize=16, fontweight='bold', pad=10)
    ax4.set_xlim(0, len(raw_features))
    ax4.set_xlabel("1280 Dimensional Graph\n(Contains Noise/Irrelevant Data)", fontsize=13)
    ax4.get_yaxis().set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    # Step 5: PSO Filtering - with CLEAR vertical callout labels!
    ax5 = plt.subplot(2, 3, 5)
    
    # Redraw faint ghost graph
    ax5.plot(x_val, raw_features, color='lightgray', lw=0.5, alpha=0.5) 
    
    top_indices = np.argsort(raw_features)[-15:][::-1]
    top_activations = raw_features[top_indices]
    
    # Clear, descriptive names
    trait_names = [
        "Leaf Shape Profiling", "Color Saturation", "Diagnostic Veining", 
        "Edge Margin Status", "Leaf Tip Geometry", "Stem Connection Angle", 
        "Upper Skin Texture", "Base Chlorophyll Levels", "Secondary Vascular Angles", 
        "Gloss & Reflection", "Blade Asymmetry Ratio", "Midrib Thickness", 
        "Micro-texture Density", "Depth / Fold Cues", "Pathogen Visual Anomalies"
    ]
    
    # Sort them purely by X axis so we can draw alternate callout heights without crossing
    sorted_pairs = sorted(zip(top_indices, top_activations, trait_names), key=lambda x: x[0])
    
    # Set y limit drastically higher to give room for vertical text boxes
    max_act = max(top_activations)
    ax5.set_ylim(0, max_act + 7.5)  
    
    # Base offset for text heights
    base_offsets = [0.8, 3.5, 6.2]
    
    for i, (idx, act, name) in enumerate(sorted_pairs):
        # 1. Draw solid red stem up to the actual activation value point
        ax5.plot([idx, idx], [0, act], color='red', lw=2.0)
        ax5.scatter(idx, act, color='red', s=40, zorder=5) # Bulb at actual value
        
        # 2. Draw a dashed, thinner line extending UP to the text label so it looks like a clear callout
        text_y_pos = act + base_offsets[i % 3] 
        ax5.plot([idx, idx], [act, text_y_pos - 0.2], color='darkred', lw=1.0, linestyle=':')
        
        # 3. Add perfectly readable vertical text
        ax5.text(idx, text_y_pos, name, rotation=90, fontsize=12, color='#990000', 
                 ha='center', va='bottom', fontweight='bold',
                 bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9))

    ax5.set_title(f'5. PSO Optimization Logic', fontsize=16, fontweight='bold', pad=10)
    ax5.set_xlim(0, len(raw_features))
    
    ax5.set_xlabel("Noise Removed | 15 Key Traits Extracted via PSO", fontsize=13, color='#b30000', fontweight='bold')
    ax5.get_yaxis().set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_visible(False)

    # Step 6: SVM Output
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.5, 0.65, "Support Vector Machine\n(SVM) Classifier", 
             ha='center', va='center', fontsize=14, fontweight='bold', color='#004d00',
             bbox=dict(boxstyle="round,pad=1.5", edgecolor='#006600', facecolor='#e6ffe6', lw=2))
    
    ax6.text(0.5, 0.25, "FINAL DECISION:\nClassified Plant Identity", 
             ha='center', va='center', fontsize=16, fontweight='bold', color='#cc0000')
    
    ax6.annotate('', xy=(0.5, 0.45), xytext=(0.5, 0.55), arrowprops=dict(arrowstyle="<-", lw=3.5, color='#333333'))
    ax6.axis('off')

    plt.tight_layout(pad=4.0)
    
    output_plot = "extraction_pipeline_process.png"
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    print(f"\nSUCCESS: Updated Graphical Process Visualization saved as '{output_plot}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="testing_dataset/1.jpg", help="Path to image")
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        visualize_pipeline(args.image)
    else:
        print(f"Error: Image {args.image} not found.")
