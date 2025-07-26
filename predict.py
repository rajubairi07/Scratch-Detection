import os
import glob
import re
import argparse
import yaml
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"ERROR: Could not read configuration file '{config_path}': {e}")
        return None

def find_latest_checkpoint(checkpoint_path):
    """
    Finds the latest model checkpoint file based on the epoch number.
    
    Args:
        checkpoint_path (str): The path pattern for checkpoints from config.yaml.
        
    Returns:
        str: The full path to the latest checkpoint file, or None if not found.
    """
    # Get the directory where checkpoints are stored
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        return None
        
    # Get the filename pattern (e.g., 'model_epoch_*.h5')
    filename_pattern = os.path.basename(checkpoint_path).replace('{epoch:02d}', '*')
    
    # Find all files matching the pattern
    files = glob.glob(os.path.join(checkpoint_dir, filename_pattern))
    
    if not files:
        return None
    
    # Extract epoch numbers and find the file with the highest one
    latest_file = max(files, key=os.path.getctime)
    return latest_file

if __name__ == '__main__':
    # --- 1. Load Configuration and Parse Arguments ---
    config = load_config()
    if config is None:
        exit()

    parser = argparse.ArgumentParser(description='Predict scratches on an image using a trained U-Net model.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image for prediction.')
    parser.add_argument('--model', type=str, default=None, help='(Optional) Path to a specific .h5 model file. Overrides automatic checkpoint loading.')
    args = parser.parse_args()

    # --- 2. Determine which model file to load ---
    MODEL_TO_LOAD = None
    
    if args.model:
        # If a model is specified via command line, use it
        MODEL_TO_LOAD = args.model
        print(f"--- Loading specified model: {MODEL_TO_LOAD} ---")
    else:
        # Otherwise, find the latest checkpoint
        print("--- Searching for the latest checkpoint... ---")
        latest_checkpoint = find_latest_checkpoint(config['paths']['checkpoint_save_path'])
        if latest_checkpoint:
            MODEL_TO_LOAD = latest_checkpoint
            print(f"--- Found and loading latest checkpoint: {MODEL_TO_LOAD} ---")
        else:
            # As a fallback, use the final model path from config
            MODEL_TO_LOAD = config['paths']['model_save_path']
            print(f"--- No checkpoints found. Loading final model: {MODEL_TO_LOAD} ---")

    if not os.path.exists(MODEL_TO_LOAD):
        print(f"ERROR: Model file not found at {MODEL_TO_LOAD}")
        exit()

    # --- 3. Load Model and Image ---
    # Load model parameters from config
    IMG_HEIGHT = config['model_params']['img_height']
    IMG_WIDTH = config['model_params']['img_width']

    # Load the trained model
    # We don't need to compile it for prediction
    model = load_model(MODEL_TO_LOAD, compile=False)

    # Load and preprocess the input image
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"ERROR: Could not read image from {args.image}")
        exit()
        
    original_shape = image.shape
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize and expand dimensions to match model input shape (1, height, width, 1)
    normalized_image = (resized_image / 255.0).astype(np.float32)
    input_tensor = np.expand_dims(np.expand_dims(normalized_image, axis=0), axis=-1)

    # --- 4. Make Prediction ---
    print("\n--- Predicting mask... ---")
    predicted_mask = model.predict(input_tensor)[0] # Get the first (and only) item in the batch

    # --- 5. Post-process and Classify ---
    min_val, max_val = np.min(predicted_mask), np.max(predicted_mask)
    print(f"DEBUG: Model prediction confidence range: min={min_val:.4f}, max={max_val:.4f}")

    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]))

    confidence_threshold = 0.1 
    pixel_threshold = int(confidence_threshold * 255)
    _, binary_mask = cv2.threshold((predicted_mask_resized * 255).astype(np.uint8), pixel_threshold, 255, cv2.THRESH_BINARY)
    
    # --- 6. Classify Image as Good or Bad ---
    # Define a threshold for how many scratch pixels count as a "bad" image
    # This helps ignore tiny noise or insignificant detections.
    scratch_pixel_threshold = 50 
    
    # Count the number of white pixels (scratches) in the binary mask
    num_scratch_pixels = cv2.countNonZero(binary_mask)
    
    print("\n--- Classification Result ---")
    if num_scratch_pixels > scratch_pixel_threshold:
        print(f"RESULT: This is a BAD image (scratches detected).")
        print(f"(Found {num_scratch_pixels} scratch pixels, which is more than the threshold of {scratch_pixel_threshold})")
    else:
        print(f"RESULT: This is a GOOD image (no significant scratches detected).")
        print(f"(Found {num_scratch_pixels} scratch pixels, which is less than the threshold of {scratch_pixel_threshold})")
    
    # --- 7. Save Output ---
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay_image[binary_mask == 255] = (0, 0, 255) # BGR format, so (0,0,255) is red

    output_filename_base = os.path.splitext(os.path.basename(args.image))[0]
    mask_output_path = f"{output_filename_base}_predicted_mask.png"
    overlay_output_path = f"{output_filename_base}_overlay.png"

    cv2.imwrite(mask_output_path, binary_mask)
    cv2.imwrite(overlay_output_path, overlay_image)

    print(f"\nPrediction complete!")
    print(f"Predicted mask saved to: {mask_output_path}")
    print(f"Overlay image saved to: {overlay_output_path}")

