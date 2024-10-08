import csv

import torch
import torchvision
from torchvision.models.video import swin3d_b, Swin3D_B_Weights

from custom_swin_3d import CustomSwin3D

# Params.
video_path = "/home/andrej/data/kinetics400/chess.mp4"
labels_path = "/home/andrej/data/kinetics400/labels/kinetics_400_labels.csv"
use_custom_swin3d = False
use_custom_transform = True
use_gpu = True
infer_in_train_mode = True

# Load labels.
with open(labels_path, mode="r") as file:
    csv_reader = csv.DictReader(file)
    labels = {row["id"]: row["name"] for row in csv_reader}

# Load Swin3D model.
if use_custom_swin3d:
    print("Using custom Swin3D model")
    model = CustomSwin3D(model_size="base", num_classes=400, use_pretrained_weights=True, use_swin_v2=False)
else:
    print("Using default Swin3D model")
    model = swin3d_b(weights="DEFAULT")

# Transformation.
transform = Swin3D_B_Weights.KINETICS400_V1.transforms()
if use_custom_transform:
    transform.crop_size = [150, 150]
    transform.resize_size = [150]

print(f"Transform type: {type(transform)}")
print("Transform:")
print(transform)

# Read video.
video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
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

# Move data to GPU.
if use_gpu:
    print("Using GPU")
    model.to("cuda")
    video = video.to("cuda")
else:
    print("Not using GPU")

# Inference.
if infer_in_train_mode:
    print("Inferring in train mode")
    model.train()
    torch.set_grad_enabled(True)
    outputs = model(video)
else:
    print("Inferring in eval mode")
    model.eval()
    torch.set_grad_enabled(False)
    outputs = model(video)

# Get the predicted class.
_, predicted = torch.max(outputs, 1)
index = predicted.item()
print(f"Predicted class: {labels[str(index)]}")
