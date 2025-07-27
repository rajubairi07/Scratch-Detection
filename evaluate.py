import os
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import your custom modules
from dataloader import load_dataset_split
from model import build_unet_model # Needed for loading the model structure

def find_latest_checkpoint(checkpoints_dir):
    """Finds the most recently created checkpoint file in a directory."""
    if not os.path.exists(checkpoints_dir) or not os.listdir(checkpoints_dir):
        return None
    
    checkpoints = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.endswith('.h5')]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def main():
    """Main function to run the evaluation."""
    # --- 1. Load Configuration ---
    print("--- Loading configuration ---")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Reads from the 'model_params' section as you correctly pointed out
    img_height = config['model_params']['img_height']
    img_width = config['model_params']['img_width']
    
    # --- 2. Load Test Data ---
    print("\n--- Loading test dataset ---")
    X_test, y_test = load_dataset_split(
        base_data_path=config['paths']['base_data_path'],
        base_mask_path=config['paths']['base_mask_path'],
        split_name='test',
        img_size=(img_width, img_height),
        augment=False, # Never augment test data
    )

    if len(X_test) == 0:
        print("\nERROR: No test data found. Exiting.")
        return

    print(f"Loaded {len(X_test)} test images.")

    # --- 3. Load Model ---
    print("\n--- Loading model from latest checkpoint ---")
    latest_checkpoint = find_latest_checkpoint(os.path.dirname(config['paths']['checkpoint_save_path']))
    
    if latest_checkpoint:
        print(f"Found latest checkpoint: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)
    else:
        print(f"WARNING: No checkpoints found. Trying to load final model from {config['paths']['model_save_path']}")
        if os.path.exists(config['paths']['model_save_path']):
             model = tf.keras.models.load_model(config['paths']['model_save_path'])
        else:
            print("\nERROR: No trained model found. Please train the model first.")
            return
            
    # --- 4. Make Predictions on the Entire Test Set ---
    print("\n--- Making predictions on the test set ---")
    predicted_masks = model.predict(X_test)

    # --- 5. Calculate Metrics ---
    print("\n--- Calculating precision and recall ---")
    
    # Get thresholds from config
    confidence_threshold = config['prediction']['confidence_threshold']
    pixel_threshold = config['prediction']['pixel_threshold_for_bad_image']
    
    true_labels = []
    predicted_labels = []

    for i in range(len(y_test)):
        # Determine the true label: 1 if 'bad' (has scratches), 0 if 'good'
        true_mask = y_test[i]
        is_true_bad = np.sum(true_mask) > 0
        true_labels.append(1 if is_true_bad else 0)

        # Determine the predicted label
        predicted_mask = predicted_masks[i]
        binary_mask = (predicted_mask > confidence_threshold).astype(np.uint8)
        num_scratch_pixels = np.sum(binary_mask)
        is_predicted_bad = num_scratch_pixels > pixel_threshold
        predicted_labels.append(1 if is_predicted_bad else 0)

    # --- 6. Print Report ---
    print("\n" + "="*50)
    print("          EVALUATION RESULTS ON TEST SET")
    print("="*50)
    
    # Note: '1' corresponds to 'bad' and '0' corresponds to 'good'
    report = classification_report(true_labels, predicted_labels, target_names=['Good (0)', 'Bad (1)'])
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
