# predict.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Constants
IMG_SIZE = 128

def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for prediction
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (original PIL Image, preprocessed numpy array)
    """
    # Load image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img, img_array

def predict_single_image(model_path, image_path, threshold=0.5):
    """
    Predict if an image is real or fake
    
    Args:
        model_path: Path to the trained model (.h5 file)
        image_path: Path to the image file
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        dict: Dictionary containing classification results
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Load and preprocess image
        img, img_array = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Determine classification and confidence
        is_real = prediction > threshold
        classification = "REAL" if is_real else "FAKE"
        confidence = prediction if is_real else 1 - prediction
        
        # Display the image with prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        
        # Set title color based on prediction
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
        plt.title(f"Prediction: {classification}\nConfidence: {confidence:.2f}", color=color, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save prediction visualization
        results_dir = os.path.dirname(os.path.abspath(model_path))
        os.makedirs(os.path.join(results_dir, 'predictions'), exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'predictions', f"{os.path.basename(image_path)}_prediction.png"))
        plt.close()
        
        # Prepare results
        results = {
            "classification": classification,
            "confidence": float(confidence),
            "raw_score": float(prediction),
            "threshold": threshold,
            "visualization_path": os.path.join(results_dir, 'predictions', f"{os.path.basename(image_path)}_prediction.png")
        }
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {
            "error": str(e),
            "classification": None,
            "confidence": None
        }

def batch_predict(model_path, image_dir, output_csv=None):
    """
    Run predictions on a directory of images
    
    Args:
        model_path: Path to the trained model (.h5 file)
        image_dir: Directory containing images to predict
        output_csv: Path to save CSV results (optional)
        
    Returns:
        list: List of prediction results
    """
    # Load model
    model = load_model(model_path)
    
    # List all image files
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if any(f.lower().endswith(ext) for ext in valid_extensions)
    ]
    
    results = []
    
    # Process each image
    for img_path in image_files:
        try:
            # Load and preprocess image
            _, img_array = load_and_preprocess_image(img_path)
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            
            # Determine classification and confidence
            is_real = prediction > 0.5
            classification = "REAL" if is_real else "FAKE"
            confidence = prediction if is_real else 1 - prediction
            
            # Store result
            result = {
                "image": os.path.basename(img_path),
                "classification": classification,
                "confidence": float(confidence),
                "raw_score": float(prediction)
            }
            
            results.append(result)
            
            print(f"Processed {os.path.basename(img_path)}: {classification} ({confidence:.2f})")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")
    
    # Save to CSV if requested
    if output_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect deepfakes in images')
    parser.add_argument('--model', required=True, help='Path to the trained model (.h5 file)')
    parser.add_argument('--image', help='Path to the image file to predict')
    parser.add_argument('--dir', help='Directory containing images to predict in batch')
    parser.add_argument('--output', help='Path to save CSV results (for batch prediction)')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image prediction
        results = predict_single_image(args.model, args.image)
        print(f"Classification: {results['classification']} with {results['confidence']:.2%} confidence")
    
    elif args.dir:
        # Batch prediction
        batch_predict(args.model, args.dir, args.output)
    
    else:
        parser.error("Either --image or --dir must be specified")