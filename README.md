<<<<<<< HEAD
# Scratch-Detection
=======
Scratch Detection using U-Net
Project Overview
This project is a deep learning solution for automatically identifying and locating scratch defects on images of text. It uses a U-Net segmentation model, built with TensorFlow and Keras, to classify each pixel in an image as either a scratch or background, generating a precise mask that highlights any damage.

Project Structure
Scratch-Detection/
├── checkpoints/              # Saved models from each epoch appear here
├── data/
│   ├── train/
│   │   ├── good/
│   │   └── bad/
│   └── test/
│       ├── good/
│       └── bad/
├── masks/
│   ├── train/              # Masks for train/bad images
│   └── test/               # Masks for test/bad images
├── config.yaml             # Main configuration file
├── dataloader.py           # Handles data loading, augmentation, and balancing
├── model.py                # Defines the U-Net architecture
├── predict.py              # Script to run predictions on new images
├── train.py                # Main script to train the model
└── requirements.txt        # List of Python dependencies

What I've Done
This project involved several key stages, from data preparation to model deployment:

Data Sourcing and Structuring: I organized the dataset into train and test folders, further subdivided into good (no defects) and bad (with scratches) categories. I created corresponding pixel-perfect masks for every "bad" image, which are essential for training the segmentation model.

Data Augmentation: To improve model robustness and prevent overfitting, I implemented an on-the-fly data augmentation pipeline using the Albumentations library. This process applies random transformations like rotations, flips, and brightness adjustments to the training data.

Handling Class Imbalance: The initial dataset had significantly more "good" images than "bad" ones. To address this, I implemented an oversampling technique in the dataloader.py script. This balances the dataset by repeating the "bad" images during training, ensuring the model doesn't become biased towards the majority class.

Model Implementation: I built the U-Net architecture from scratch in model.py. This model is specifically designed for segmentation and excels at capturing both contextual information and precise location details.

Configuration Management: To make the project maintainable, I used a config.yaml file to manage all important parameters, including file paths, image dimensions, and training settings like epochs and batch size.

Training and Checkpointing: The main train.py script orchestrates the training process. I implemented a ModelCheckpoint callback to save the model's state after every epoch, which is useful for tracking progress and resuming training.

Prediction Script: I developed a predict.py script that loads the trained model, processes a new input image, and generates two outputs:

A final classification of the image as "GOOD" or "BAD".

An overlay image that visually highlights the detected scratches in red.

Technical Stack
Model: U-Net (TensorFlow/Keras)

Image Processing: OpenCV

Data Augmentation: Albumentations

Configuration: PyYAML

How to Run This Project
1. Set Up Your Environment
# Navigate to the project folder
cd /path/to/Scratch-Detection

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate


2. Install Dependencies
pip install -r requirements.txt


3. Configure Paths
Open config.yaml and update base_data_path and base_mask_path to the correct absolute paths for your machine.

4. Train the Model
python train.py


5. Make a Prediction
python predict.py --image "C:/path/to/your/new_image.png"
>>>>>>> feded78 (Initial commit: U-Net scratch detection project setup)
