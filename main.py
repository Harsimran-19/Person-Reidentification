import torch
import cv2
import os
import time
import numpy as np
from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.transforms as transforms
from utils import ResNet50Model,annotate


h, w = 256, 128
transform_test = transforms.Compose([
    transforms.Resize((h, w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

np.random.seed(41)

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
COLORS = np.random.randint(0, 255, size=(1, 3))


model = ResNet50Model(class_num=751)
model.load_state_dict(torch.load('ReID_20.pth'))
model.to(device)
model.eval()


tracker = DeepSort(max_age=30, embedder=model)

VIDEO_PATH = 'sample.mp4'
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]

out = cv2.VideoWriter(
    f"{OUT_DIR}/out.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, 
    (frame_width, frame_height)
)

frame_count = 0
total_fps = 0
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    if ret:
        resized_frame = transform_test(frame)
        frame_tensor = resized_frame.to(device)

        start_time = time.time()
        with torch.no_grad():
            embeddings = model(frame_tensor.unsqueeze(0))

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        total_fps += fps
        frame_count += 1

        print(f"Frame {frame_count}/{frames}")
        track_start_time = time.time()
        tracks = tracker.update_tracks(embeddings, frame=frame)
        track_end_time = time.time()
        track_fps = 1 / (track_end_time - track_start_time)
        if len(tracks) > 0:
            frame = annotate(
                tracks, 
                frame, 
                resized_frame,
                frame_width,
                frame_height,
                COLORS
            )
        # cv2.putText(
        #     frame,
        #     f"FPS: {fps:.1f}",
        #     (int(20), int(40)),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1,
        #     color=(0, 0, 255),
        #     thickness=2,
        #     lineType=cv2.LINE_AA
        # )
        out.write(frame)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
