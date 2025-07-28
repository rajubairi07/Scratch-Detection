import os
import cv2
import numpy as np
import albumentations as A
import random

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2)
])

def load_dataset_split(base_data_path, base_mask_path, split_name, img_size=(128, 128), augment=False):
    """
    Loads a specific dataset split (train or test) from structured folders.
    
    Args:
        base_data_path (str): Path to the main data directory containing train/test folders.
        base_mask_path (str): Path to the main mask directory containing train/test folders.
        split_name (str): The name of the split to load, e.g., 'train' or 'test'.
        img_size (tuple): The target size to resize images to.
        augment (bool): If True, applies data augmentation to the images and masks.

    Returns:
        tuple: A tuple containing two numpy arrays (X, y) for images and masks.
    """
    X = []
    y = []

    good_images_dir = os.path.join(base_data_path, split_name, 'good')
    bad_images_dir = os.path.join(base_data_path, split_name, 'bad')
    mask_images_dir = os.path.join(base_mask_path, split_name)

    good_image_paths = [os.path.join(good_images_dir, fname) for fname in os.listdir(good_images_dir)]
    bad_image_paths = [os.path.join(bad_images_dir, fname) for fname in os.listdir(bad_images_dir)]

    # --- CLASS IMBALANCE HANDLING (OVERSAMPLING) ---
    # This is only applied to the training set (when augment=True)
    if augment and len(good_image_paths) > 0 and len(bad_image_paths) > 0:
        imbalance_ratio = len(good_image_paths) // len(bad_image_paths)
        if imbalance_ratio > 1:
            print(f"INFO: Class Imbalance Detected in '{split_name}' set.")
            print(f"       Good images: {len(good_image_paths)}, Bad images: {len(bad_image_paths)}")
            print(f"       Oversampling bad images by a factor of {imbalance_ratio}.")
            bad_image_paths = bad_image_paths * imbalance_ratio
            print(f"       New bad image count (for training): {len(bad_image_paths)}")


    all_image_paths = good_image_paths + bad_image_paths
    random.shuffle(all_image_paths)

    for img_path in all_image_paths:
        try:
            # Load image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, img_size)

            mask = np.zeros(img_size, dtype=np.float32)

            # Check if the image is from the 'bad' directory
            if os.path.basename(os.path.dirname(img_path)) == 'bad':
                # If it's a bad image, find and load its corresponding mask
                mask_path = os.path.join(mask_images_dir, os.path.basename(img_path))
                if os.path.exists(mask_path):
                    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.resize(mask_image, img_size)
                else:
                    print(f"Warning: Mask not found for {os.path.basename(img_path)} in {mask_images_dir}")
                    continue

            # Apply augmentation if flag is set
            if augment:
                augmented = transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # Normalize and add to lists
            X.append(image / 255.0)
            y.append(mask / 255.0)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    # Convert lists to numpy arrays and add channel dimension
    X = np.array(X, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)
    y = np.array(y, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)

    return X, y
