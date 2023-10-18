# Person Reidentification

![Reidentification](./Images/dataset.jpg)


## Table of Contents

1. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
2. [Person Detection and Tracking](#person-detection-and-tracking)
3. [Feature Extraction](#feature-extraction)
4. [Person Re-Identification Model](#person-re-identification-model)
5. [Visualization and Demonstration](#visualization-and-demonstration)

## Data Collection and Preprocessing

### 1. Data Collection
- Collected publicly available footage capturing people walking.

### 2. Data Preprocessing
- Extracted frames from the video data using OpenCV.

## Person Detection and Tracking

### 1. Person Detection
- Implemented person detection using a pre-trained Faster RCNN.

### 2. Person Tracking
- Developed a tracking algorithm using DeepSORT to track individuals.

## Feature Extraction

### 1. Feature Extraction
- Utilized a ResNet model trained on the Market-1501 dataset for feature extraction.

## Person Re-Identification Model

### 1. Model Implementation
- Designed and implemented a person re-identification model using PyTorch.

### 2. Model Evaluation
- The model results after 20 epochs were :
- Train Loss: 0.1494 Train Acc: 0.9925
- Val Loss: 0.5166 Val Acc: 0.8855
- 
## Visualization and Demonstration

### 1. Visualizations
- Created visualization with opencv to showcase the effectiveness of the person re-identification model.


