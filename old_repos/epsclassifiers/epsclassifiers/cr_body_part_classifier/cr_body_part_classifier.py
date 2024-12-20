import torch, pydicom, os
import matplotlib.pyplot as plt
import torchxrayvision as xrv
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from epsdatasets.helpers.gradient_mimic.gradient_mimic_dataset_helper import GradientMimicDatasetHelper


def preprocess_dicom(dicom_data):
    """Convert DICOM to normalized image array"""
    # Extract the pixel array
    image = dicom_data.pixel_array

    # Normalize based on window center and width if available
    if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        vmin = center - width // 2
        vmax = center + width // 2
        image = np.clip(image, vmin, vmax)

    # Normalize to 0-255 range
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    return Image.fromarray(image)


class CrBodyPartClassifier:
    def __init__(self):
        """Initialize classifier using TorchXRayVision's DenseNet121"""
        # Load the pretrained model
        self.base_model = xrv.models.DenseNet(weights="densenet121-res224-all")

        # Create feature extractor (keeping densenet features)
        self.feature_extractor = self.base_model.features

        # Freeze the feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Create new classifier for 1 class with proper dimensionality
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # Force 1x1 spatial dimensions
            torch.nn.Flatten(),  # Convert to 1D
            torch.nn.Linear(1024, 512),  # DenseNet121 outputs 1024 features
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1)  # No sigmoid here
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.classifier = self.classifier.to(self.device)

        # Transform for single-channel images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)  # Should output [batch_size, 1024, 7, 7]
        # Pass through classifier
        return self.classifier(features)  # Raw logits

    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
        """Train the model with checkpoint saving/loading"""
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Try to load checkpoint
        start_epoch = self.load_checkpoint(optimizer)

        for epoch in range(start_epoch, num_epochs):
            # Training phase
            self.classifier.train()
            running_train_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # Validation phase
            self.classifier.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch, optimizer, avg_train_loss, avg_val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}\n')

    def fine_tune(self, train_loader, val_loader, learning_rate=0.00001, num_epochs=30):
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # Training phase
            self.classifier.train()
            running_train_loss = 0.0

            for inputs, labels in tqdm(train_loader, total=len(train_loader), desc="Training"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                torch.save({"inputs": inputs, "labels": labels}, "current_training_data.pt")

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            # Validation phase
            self.classifier.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(val_loader)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}\n')

    def predict(self, dicom_path, threshold=0.5):
        """Predict multiple labels for X-ray DICOM image"""
        self.base_model.eval()

        # Load and preprocess DICOM
        dicom_data = pydicom.dcmread(dicom_path)
        image = preprocess_dicom(dicom_data)
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.base_model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).float()

        predicted_classes = torch.nonzero(predictions[0]).cpu().numpy().flatten()
        confidences = probabilities[0][predicted_classes].cpu().numpy()

        return list(zip(predicted_classes, confidences))

    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.feature_extractor.eval()
        self.classifier.eval()

        all_preds = []
        all_labels = []
        running_val_loss = 0.0
        criterion = torch.nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                # Apply sigmoid here for predictions
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()

                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Convert lists to numpy arrays properly
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)

        # Calculate metrics using sample-based averaging
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)

        print(f"\nValidation Metrics:")
        print(f"Validation Loss: {running_val_loss/len(val_loader):.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

    def save_checkpoint(self, epoch, optimizer, train_loss, val_loss, path='~/kedar/checkpoints/bodyPart/checkpoint.pth'):
        """Save model checkpoint to disk"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, path)

    def load_checkpoint(self, optimizer, path='checkpoint.pth'):
        """Load model checkpoint from disk"""
        if not os.path.exists(path):
            return 0  # Start from epoch 0 if no checkpoint

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1  # Return next epoch to start from


def main():
    # Initialize the classifier
    classifier = CrBodyPartClassifier()

    # Setup data loaders
    helper = GradientMimicDatasetHelper(gradient_data_gcs_bucket_name="gradient-crs",
                                        gradient_data_gcs_dir="16AG02924",
                                        gradient_images_gcs_bucket_name="epsilon-data-us-central1",
                                        gradient_images_gcs_dir="GRADIENT-DATABASE/CR/16AG02924",
                                        mimic_gcs_bucket_name="epsilonlabs-filestore",
                                        mimic_gcs_dir="mimic2-dicom/mimic-cxr-jpg-2.1.0.physionet.org",
                                        exclude_file_name="/home/andrej/work/epsclassifiers/epsclassifiers/cr_body_part_classifier/chest_scan_results.txt",
                                        seed=42)
    train_loader = helper.get_torch_train_data_loader(batch_size=16, num_workers=4)
    val_loader = helper.get_torch_validation_data_loader(batch_size=16, num_workers=4)

    # Fine-tune the model
    classifier.fine_tune(train_loader, val_loader, learning_rate=0.001)

    # Evaluate the model
    classifier.evaluate(val_loader)


if __name__ == '__main__':
    main()
