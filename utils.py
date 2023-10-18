import cv2
import numpy as np

# Define a function to convert detections to SORT format.
def convert_detections(detections, threshold, classes):
    # Get the bounding boxes, labels and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    # Filter out low confidence scores and non-person classes.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for i, box in enumerate(boxes):
        # Append ([x, y, w, h], score, label_string).
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(labels[i])
            )
        )

    return final_boxes

# Function for bounding box and ID annotation.
def annotate(tracks, frame, resized_frame, frame_width, frame_height, colors):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_class = track.det_class
        x1, y1, x2, y2 = track.to_ltrb()
        p1 = (int(x1/resized_frame.shape[1]*frame_width), int(y1/resized_frame.shape[0]*frame_height))
        p2 = (int(x2/resized_frame.shape[1]*frame_width), int(y2/resized_frame.shape[0]*frame_height))
        # Annotate boxes.
        color = colors[int(track_class)]
        cv2.rectangle(
            frame, 
            p1, 
            p2, 
            color=(int(color[0]), int(color[1]), int(color[2])), 
            thickness=2
        )
        # Annotate ID.
        cv2.putText(
            frame, f"ID: {track_id}", 
            (p1[0], p1[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2,
            lineType=cv2.LINE_AA
        )
    return frame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

class ResNet50Model(nn.Module):
    def __init__(self, class_num):
        super(ResNet50Model, self).__init__()
        # Load the model
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

