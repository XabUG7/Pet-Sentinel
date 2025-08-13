# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 01:53:06 2025

@author: XabUG7
"""

import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from ultralytics import YOLO
import pickle

import warnings
warnings.filterwarnings("ignore")



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------- MODEL ---------
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
    
    
# precompute values and models to save time when running is_my_pet function
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(-1, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(-1, 1, 1)

with open("assets/embedding gallery/my_cat_gallery.pkl", "rb") as f:
    cat_gallery = pickle.load(f)

with open("assets/embedding gallery/my_dog_gallery.pkl", "rb") as f:
    dog_gallery = pickle.load(f)

# Load the models with saved weights
cat_model = EmbeddingNet()
cat_model.load_state_dict(torch.load("assets/trained models/cat_model.pt"))
cat_model.eval()
cat_model.to(DEVICE)

dog_model = EmbeddingNet()
dog_model.load_state_dict(torch.load("assets/trained models/dog_model.pt"))
dog_model.eval()
dog_model.to(DEVICE)


def is_my_pet(img_crop, animal, threshold=0.85):

    if animal == "cat":
        model = cat_model
        gallery = cat_gallery
    elif animal == "dog":
        model = dog_model
        gallery = dog_gallery

    # Convert to RGB and resize
    rgb_image = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)  
    rgb_image = cv2.resize(rgb_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

    # rgb_tensor = img_crop # remove this for production
    rgb_tensor = rgb_tensor.to(DEVICE)


    # Normalize
    rgb_tensor = (rgb_tensor - mean) / std


    rgb_tensor = rgb_tensor.unsqueeze(0)
    with torch.no_grad():
        emb = model(rgb_tensor).cpu().numpy().flatten()
    score = np.mean(cosine_similarity([emb], gallery))

    if score >= threshold:
        print(f"Your {animal} detected with a score: {score}")
        return True
    else:
        return False
    
    

# YOLOv8n: Already trained with cat & dog (Index 15:cat, Index 16:dog)
# https://docs.ultralytics.com/datasets/detect/coco/#coco-pretrained-models
model = YOLO('yolov8n.pt')

names_dict = model.model.names
cat_id = [k for k, v in names_dict.items() if v == 'cat'][0]
dog_id = [k for k, v in names_dict.items() if v == 'dog'][0]
person_id = [k for k, v in names_dict.items() if v == 'person'][0]
bear_id = [k for k, v in names_dict.items() if v == 'bear'][0]


# Select here every how many frames you want to run image recognition
run_ever_n_frames = 1
threshold_cat = 0.25
threshold_dog = 0.4

def main():
    # Opening webcam
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        frame_count += 1
        
        # Utilizing YOLO model detecting (confidence level = 30%)
        results = model.predict(frame, conf=0.3, verbose=False)
    
        my_cat_detected = False
        my_dog_detected = False
        reject_flag = False
    
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
    
            # Color
            if label == 'cat':
                # run here the image recognition model
                if frame_count >= run_ever_n_frames:
                    my_cat_detected = is_my_pet(frame[y1:y2, x1:x2], label, threshold_cat)
                    if my_cat_detected:
                        color = (0, 255, 0)    # Green
                    else:
                        color = (0, 0, 255)    # Red
                    frame_count = 0
                    
    
            elif label == 'dog':
                # run here the image recognition model
                if frame_count >= run_ever_n_frames:
                    my_dog_detected = is_my_pet(frame[y1:y2, x1:x2], label, threshold_dog)
                    if my_dog_detected:
                        color = (0, 255, 0)    # Green
                    else:
                        color = (0, 0, 255)    # Red
                    frame_count = 0
    
            elif label == 'person':
                color = (0, 0, 255)    # Red
                reject_flag = True
            elif label == 'bear':
                color = (0, 0, 255)  # Red
                reject_flag = True
            else:
                continue
                # color = (0, 255, 255)    # Yellow
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
        # Decision Logic is shwon as text
        # in real environment it would trigger gate controls
        if reject_flag and not my_cat_detected and not my_dog_detected:
            status = "REJECT (person, bear or unknown at the gate)"
        elif my_cat_detected or my_dog_detected:
            status = "Gate Open!"

            
        else:
            status = "EMPTY or not allowed objects in screen"
    
        # Status on Screen
        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3)
        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    
        cv2.imshow('Pet Gate', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
