import os
import yaml
from tensorflow.keras.callbacks import ModelCheckpoint
from dataloader import load_dataset_split # Corrected import
from model import build_unet_model

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"ERROR: Could not read configuration file '{config_path}': {e}")
        return None

if __name__ == '__main__':
    # --- Load Configuration ---
    config = load_config()
    if config is None:
        exit()

    # Model and training parameters from config
    IMG_HEIGHT = config['model_params']['img_height']
    IMG_WIDTH = config['model_params']['img_width']
    EPOCHS = config['training_params']['epochs']
    BATCH_SIZE = config['training_params']['batch_size']
    INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)

    # File paths from config
    BASE_DATA_PATH = config['paths']['base_data_path']
    BASE_MASK_PATH = config['paths']['base_mask_path']
    MODEL_SAVE_PATH = config['paths']['model_save_path']
    CHECKPOINT_SAVE_PATH = config['paths']['checkpoint_save_path']

    # --- 1. Load Training Data ---
    # Using the corrected function name: load_dataset_split
    print("--- Loading Training Data ---")
    X_train, y_train = load_dataset_split(
        base_data_path=BASE_DATA_PATH,
        base_mask_path=BASE_MASK_PATH,
        split_name='train',
        img_size=(IMG_WIDTH, IMG_HEIGHT),
        augment=True # Augment the training data and handle class imbalance
    )

    # --- 2. Load Testing Data ---
    # Using the corrected function name: load_dataset_split
    print("\n--- Loading Testing Data ---")
    X_test, y_test = load_dataset_split(
        base_data_path=BASE_DATA_PATH,
        base_mask_path=BASE_MASK_PATH,
        split_name='test',
        img_size=(IMG_WIDTH, IMG_HEIGHT),
        augment=False # Do not augment the test data
    )

    # --- 3. Build Model ---
    print("\n--- Building U-Net Model ---")
    model = build_unet_model(INPUT_SHAPE)
    model.summary()

    # --- 4. Train Model ---
    if len(X_train) > 0 and len(X_test) > 0:
        # Create checkpoint callback
        os.makedirs(os.path.dirname(CHECKPOINT_SAVE_PATH), exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            filepath=CHECKPOINT_SAVE_PATH,
            save_weights_only=False,
            save_best_only=False, # Set to False to save after every epoch
            verbose=1
        )

        print("\n--- Starting Training ---")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_callback] # Add the callback here
        )
        # --- 5. Save Final Model ---
        print(f"\n--- Training Complete. Saving final model to {MODEL_SAVE_PATH} ---")
        model.save(MODEL_SAVE_PATH)
        print("Model saved successfully!")
    else:
        print("\nERROR: Data was not loaded correctly for training or testing. Please check paths and data folders.")

