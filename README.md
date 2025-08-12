# Pet Sentinel: Personalized Pet Recognition Gate

### Overview
Pet Sentinel is an intelligent computer vision solution that uses deep learning and object detection for real-time recognition of your specific pets through a live camera feed. The current repository is trained for cat and dog recognition but the techniques can be extrapolated to more animals. This system can distinguish your pet from others, triggering actions (like opening a gate initially but can be extended to other purposes) only for your accompanied animals, while rejecting other objects, people, and animals.

The project leverages the following technologies and resources:

- Google Drive API for data acquisition
- PyTorch and torchvision for deep learning model training and inference
- Pretrained MobileNetV2 as an embedding extractor for siamese networks
- AWS was used to leverage the power of cloud computing for scalable model training
- YOLOv8 for robust object detection (cats, dogs, people, bears)
- Real-time webcam via OpenCV
- Dataset balancing, augmentation, and result verification
- Easy configuration for "cat" or "dog" recognition
- Modular reusable code base for training, evaluation, and deployment

# Table of Contents

- [Project Structure](#project-structure)
- [Setup and Dependencies](#setup-and-dependencies)
- [Workflow](#workflow)
- [Usage](#usage)
  - [How to test the app](#5.-live-webcam-demo)
- [Configuration](#configuration)
- [Dataset](#Dataset)
- [License](#License)

# Project Structure

```text
.
├── assets/
│   ├── raw dataset/
│   │   └── ... (H5 files with raw extracted tensors from Google Drive images)
│   ├── transformed data/
│   │   └── ... (Pickle files with processed tensors (uniform image size) and labels)
│   ├── embedding gallery/
│   │   └── ... (Precomputed feature vectors for images with positive label)
│   └── trained models/
│       └── ... (*.pt weights files)
├── pet-sentinel-app.py                                                    # Main real-time inference script
├── Get pictures from Google drive and convert them to tensors.ipynb       # Google Drive download and preprocessing workflow
├── Create test and train datasets.ipynb                                   # Test and train dataset creation
├── training.ipynb                                                         # Model and training code
├── evaluation.ipynb                                                       # Model evaluation code
├── requirements.txt                                                       # Python dependencies
└── README.md                                                              # Project documentation
```

# Setup and Dependencies

## Hardware

- AWS ml.p3.2xlarge instance in SageMaker for model training
- CUDA-enabled GPU recommended for training and fast inference.
- A webcam for live demonstration.

## Software

### Python Packages

- torch
- torchvision
- tqdm
- opencv-python
- pillow
- pillow-heif
- h5py
- numpy
- google-api-python-client
- google-auth
- google-auth-oauthlib
- scikit-learn
- ultralytics (YOLOv8)
- matplotlib

Install all requirements:

```bash
pip install -r requirements.txt
```

### Google Drive API

- Place your downloaded `cloud_client.json` in the project root.
- At first run, an authentication browser flow will initiate and generate `token.json`.
  (Create a Google Cloud project and Set up API access permissions)

# Workflow

1. **Data Acquisition**
    - Automatically fetches shared folders from Google Drive annotated for cat/dog images and videos (e.g., "random cat", "same cat", etc.).
2. **Preprocessing**
    - Extracts images or video frames, converts them as PyTorch tensors.
    - Writes tensors/labels to H5 files, maintaining class labels.
3. **Transformation & Dataset Splitting**
    - Normalizes tensors, resizes to uniform shape, stores as pickle for fast access.
    - Creates balanced train/test splits.
4. **Training**
    - Trains a Siamese/Embedding network (MobileNetV2 backbone) with a contrastive loss, using positive/negative pairs of your pet vs. others.
    - Model files saved for cats and dogs separately.
5. **Embedding Gallery Creation**
    - Embeddings for "your" pet from training data are saved as a feature gallery.
6. **Verification & Evaluation**
    - Verifies new unseen samples using cosine similarity against the embedding gallery, evaluates with precision/recall.
7. **Live Webcam Inference**
    - Uses YOLOv8 for all-object detection.
    - For pets, crops and embeds the region, compares against embedding gallery to allow/deny access in real-time.

# Usage

**1. Data Processing (from Google Drive)**

Ensure you have access to the labeled Google Drive folders. Follow Google Cloud documentation to generate client credentials in a json file. Place your `cloud_client.json` at the root.


**2. Dataset Normalization & Splitting**

Resize, normalize, and structure the data for model training:

```bash
Create test and train datasets.ipynb    
```

This script processes H5 files, saves normalized tensors, and splits into balanced train/test sets (as pickles).

**3. Model Training**

Train Siamese network for cats or dogs (edit script for animal type):

```bash
training.ipynb
```

Saves best embedding model weights to `assets/trained models/`.

**4. Precompute Embedding Galleries**

After training, save embeddings for all of "your" pets for verification:

```python
# evaluation.ipynb contains:
save_gallery_embeddings(model, "assets/transformed data/train_data_cat.pkl", "cat")
save_gallery_embeddings(model, "assets/transformed data/train_data_dog.pkl", "dog")
```

## 5. Live Webcam Demo

Run the real-time pet recognition gate:

```bash
python pet-sentinel-app.py
```

- Opens webcam window
- Displays bounding boxes (green for your pet, red for others)
- Shows "Gate Open!" only for recognized pets
- Press `q` to quit

# Configuration

- Set detection thresholds (`threshold_cat`, `threshold_dog`) and YOLO confidence in `pet-sentinel-app.py` as needed.


# Dataset

This project uses data from 3 different sources for training the model:

- Random dog pictures are from the following site: https://dog.ceo/dog-api/documentation/random  
  The images are fetched with "Dog Pics API Fetch.py" in the folder "Dog Pics"  
  This dataset is used without commercial purpose in this project.
- Random cat images have been fetched from the following website: https://aleatori.cat/  
  For the exact same images used during training contact us.
  This dataset is used without commercial purpose in this project.
- Own cat and dog images for training on positive recognition label. These pet images are not available online and belong to the pet owners.  
  In case you want access to the images feel free to contact us through [open an issue](https://github.com/XabUG7/Pet-Sentinel/issues)


# License

This repository is intended for personal, educational, and non-commercial use only.  
You may use, modify, and share the code and resources for research or learning purposes.  
**Commercial use, redistribution, or integration into proprietary systems is strictly prohibited** without prior written permission from the author.

If you have questions about licensing or wish to request an exception, please contact us via [open an issue](https://github.com/XabUG7/Pet-Sentinel/issues)

