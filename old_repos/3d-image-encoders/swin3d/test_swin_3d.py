import csv

import torch
import torchvision.io as io

from custom_swin_3d import CustomSwin3D

#video_path = "/home/ec2-user/data/kinetics400/-0Li7rc78jQ.mp4"
video_path = "/home/ec2-user/data/kinetics400/_NnV0Wjzq5o.mp4"
labels_path = "/home/ec2-user/data/kinetics400/labels/kinetics_400_labels.csv"

# Load labels.
with open(labels_path, mode="r") as file:
    csv_reader = csv.DictReader(file)
    labels = {row["id"]: row["name"] for row in csv_reader}

# Load Swin3D model.
model = CustomSwin3D(model_size="tiny", num_classes=400, use_pretrained_weights=True)
model.eval()

# Use the built-in transform.
transform = model.get_transform()

# Read video.
video, _, _ = io.read_video(video_path, pts_unit="sec")
print(f"Video type: {type(video)}")
print(f"Original video shape: {video.shape}")

# Video tensor is of shape (T, H, W, C), where T is number of frames (T = time), H is image height, W is image width and
# C is number of channels. We need to change it to (T, C, H, W) format.
video = video.permute(0, 3, 1, 2)
print(f"Video shape after permutation: {video.shape}")

# Preprocess the video frames.
video = transform(video)
print(f"Video shape after transform: {video.shape}")

# Add batch dimension.
video = video.unsqueeze(0)
print(f"Video shape after unsqueeze: {video.shape}")

# Perform inference.
with torch.no_grad():
    outputs = model(video)

# Get the predicted class.
_, predicted = torch.max(outputs, 1)
index = predicted.item()
print(f"Predicted class: {labels[str(index)]}")
