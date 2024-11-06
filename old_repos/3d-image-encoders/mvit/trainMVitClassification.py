import torch, logging, os, ast, SimpleITK as sitk, pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.models.video import mvit_v2_s
import torch.nn.functional as F
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class CTDataset(Dataset):
    def __init__(self, csv_path, data_dir, num_classes, transform=None, split=None, target_frames=112):
        self.data_dir = Path(data_dir)
        self.num_classes = num_classes
        self.transform = transform
        self.target_frames = target_frames
        df = pd.read_csv(csv_path)
        
        # Filter by split if provideds
        if split:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            volume_info = ast.literal_eval(row['volume'])
            nifti_file = volume_info['nifti_file']
            label = ast.literal_eval(row['label'])
            
            # Load CT volume
            sitk_image = sitk.ReadImage(os.path.join(self.data_dir, nifti_file))
            sitk_image_array = sitk.GetArrayFromImage(sitk_image)
            
            if sitk_image_array.size == 0:
                raise ValueError(f"Empty volume for file {nifti_file}")
                
            # Normalize to [0, 1]
            min_val = sitk_image_array.min()
            max_val = sitk_image_array.max()
            if max_val - min_val == 0:
                raise ValueError(f"Zero range in normalization for file {nifti_file}")
            sitk_image_array = (sitk_image_array - min_val) / (max_val - min_val)
            
            # Stack frames and keep single channel
            if self.transform:
                frames = torch.stack([self.transform(frame.astype(np.float32)) for frame in sitk_image_array])
            else:
                frames = torch.FloatTensor(sitk_image_array)
                frames = frames.unsqueeze(1)  # Add single channel dimension
            
            # Reshape to (1, T, H, W) format
            frames = frames.permute(1, 0, 2, 3)
            
            # Resize temporal dimension
            if frames.shape[1] != self.target_frames:
                frames = F.interpolate(
                    frames.unsqueeze(0),
                    size=(self.target_frames, 224, 224),  # (T, H, W)
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
            
            return frames, torch.FloatTensor(label)
        except Exception as e:
            logging.error(f"Error processing index {idx}: {str(e)}")
            raise

def pad_or_crop_spatial(image, target_size=224):
    """Center pad or crop image to target size"""
    curr_h, curr_w = image.shape[-2:]
    
    if curr_h < target_size or curr_w < target_size:
        # Calculate padding
        pad_h = max(0, (target_size - curr_h) // 2)
        pad_w = max(0, (target_size - curr_w) // 2)
        pad_h_end = target_size - curr_h - pad_h
        pad_w_end = target_size - curr_w - pad_w
        
        # Pad with zeros
        padding = ((0, 0), (pad_h, pad_h_end), (pad_w, pad_w_end))
        image = np.pad(image, padding, mode='constant', constant_values=0)
    
    elif curr_h > target_size or curr_w > target_size:
        # Center crop
        start_h = (curr_h - target_size) // 2
        start_w = (curr_w - target_size) // 2
        image = image[..., start_h:start_h+target_size, start_w+target_size]
    
    return image

def pad_or_crop_temporal(volume, target_slices=112):
    """Center pad or crop volume temporal dimension"""
    curr_slices = volume.shape[0]
    
    if curr_slices < target_slices:
        # Calculate padding
        pad_start = (target_slices - curr_slices) // 2
        pad_end = target_slices - curr_slices - pad_start
        padding = ((pad_start, pad_end), (0, 0), (0, 0))
        volume = np.pad(volume, padding, mode='constant', constant_values=0)
    
    elif curr_slices > target_slices:
        # Center crop
        start_slice = (curr_slices - target_slices) // 2
        volume = volume[start_slice:start_slice+target_slices]
    
    return volume

def preprocess_volume(volume):
    """Convert single channel 3D volume to match MViT input requirements"""
    # Convert [B, 1, D, H, W] to [B, 3, D, H, W]
    if volume.shape[1] == 1:
        volume = volume.repeat(1, 3, 1, 1, 1)
    return volume

def train(model, dataloader, criterion, optimizer, device, 
          accumulation_steps=8, val_dataloader=None, val_steps=10000):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predicted_labels = []
    error_count = 0
    optimizer.zero_grad()  # Initialize gradients to zero
    global_step = 0  # Track the global step

    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Training')):
        try:
            # Preprocess volumes
            inputs = preprocess_volume(inputs)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Scale the loss by accumulation steps
            loss = loss / accumulation_steps 

            # Backward pass to accumulate gradients
            loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps  # Multiply back to get the actual loss

            # Accumulate predictions and targets for the whole epoch
            predicted_labels = torch.sigmoid(outputs) > 0.5
            all_targets.extend(targets.cpu().numpy())
            all_predicted_labels.extend(predicted_labels.cpu().numpy())

            # Log batch metrics (optional)
            if batch_idx % 10 == 0:
                wandb.log({
                    "batch/train_loss": loss.item() * accumulation_steps,
                    "global_step": global_step
                })

            global_step += 1  # Increment global step -- avoid starting with validation

            # Validation every val_steps
            if val_dataloader and global_step % val_steps == 0:
                val_loss, val_f1, val_acc = validate(model, val_dataloader, criterion, device)
                logging.info(
                    f'Step {global_step} - '
                    f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.2f}, Val Acc: {val_acc:.2f}%'
                )

        except RuntimeError as e:
            error_count += 1
            logging.error(f"Runtime error in batch {batch_idx}: {str(e)}")
            logging.error(f"Input shape: {inputs.shape if 'inputs' in locals() else 'unknown'}")
            continue
        except Exception as e: 
            error_count += 1
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue

    if len(all_targets) == 0:
        logging.warning("No samples were successfully processed in training epoch")
        return float('inf'), 0.0, 0.0

    logging.info(f"Training completed with {error_count} errors out of {len(dataloader)} batches")

    # Calculate metrics for the whole epoch
    all_targets = np.array(all_targets)
    all_predicted_labels = np.array(all_predicted_labels)

    f1 = f1_score(all_targets, all_predicted_labels, average='micro')
    accuracy = (all_predicted_labels == all_targets).mean()

    metrics = {
        "epoch/train_loss": running_loss / (len(dataloader) - error_count),
        "epoch/train_f1": f1,
        "epoch/train_accuracy": accuracy,
        "epoch/train_error_rate": error_count / len(dataloader)
    }
    wandb.log(metrics)
    return metrics["epoch/train_loss"], metrics["epoch/train_f1"], metrics["epoch/train_accuracy"]

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predicted_labels = [] 
    error_count = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc='Validation')):
            try:
                # Preprocess volumes
                inputs = preprocess_volume(inputs)
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()

                # Accumulate predictions and targets for the whole epoch
                predicted_labels = torch.sigmoid(outputs) > 0.5
                all_targets.extend(targets.cpu().numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

            except RuntimeError as e:
                error_count += 1
                logging.error(f"Runtime error in validation batch {batch_idx}: {str(e)}")
                logging.error(f"Input shape: {inputs.shape if 'inputs' in locals() else 'unknown'}")
                continue
            except Exception as e:
                error_count += 1
                logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                continue

    if len(all_targets) == 0:
        logging.warning("No samples were successfully processed in validation")
        return float('inf'), 0.0, 0.0

    logging.info(f"Validation completed with {error_count} errors out of {len(dataloader)} batches")

    # Calculate metrics for the whole epoch
    all_targets = np.array(all_targets)
    all_predicted_labels = np.array(all_predicted_labels)

    f1 = f1_score(all_targets, all_predicted_labels, average='micro')
    accuracy = (all_predicted_labels == all_targets).mean()

    metrics = {
        "epoch/val_loss": running_loss / (len(dataloader) - error_count),
        "epoch/val_f1": f1,
        "epoch/val_accuracy": accuracy,
        "epoch/val_error_rate": error_count / len(dataloader)
    }
    wandb.log(metrics)
    return metrics["epoch/val_loss"], metrics["epoch/val_f1"], metrics["epoch/val_accuracy"]

def get_latest_checkpoint(checkpoint_dir):
    """Get latest checkpoint from directory"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob('model_*.pth'))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    return latest

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    path = checkpoint_dir / f'model_{epoch:03d}.pth'
    torch.save(checkpoint, path)
    
    # Log model checkpoint to wandb
    wandb.save(str(path))
    return path

def create_model(num_classes, device):
    # Load MViTv2-Small model
    model = mvit_v2_s(pretrained=False, num_classes=num_classes)
    print(model)
        
    return model.to(device)

def main():
    # Configuration
    config = {
        "num_classes": 22,
        "batch_size": 1,
        "num_epochs": 100,
        "learning_rate": 3e-4,
        "early_stopping_patience": 15,
        "architecture": "MViT-v2-Small",
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau"
    }
    
    # Initialize wandb
    wandb.init(
        project="ct-class-mvit-224-224-112",
        config=config,
        name=f"mvit-v2-run-{wandb.util.generate_id()}",
        dir=str(Path('~/kedar/training-logs/mvit-large/wandb').expanduser())
    )
    
    # Setup checkpoint directory
    checkpoint_dir = Path('~/kedar/training-logs/mvit-large/checkpoints').expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = checkpoint_dir.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / 'training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45], std=[0.225])
    ])
    
    # Create datasets using CSV split column
    csv_path = '/mnt/gradient-cts-nifti/ct_chest_training_sample.csv'
    data_dir = '/mnt/gradient-cts-nifti/16AGO2024/'
    
    train_dataset = CTDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        num_classes=config["num_classes"],
        transform=transform,
        split='train',  # Add split parameter to CTDataset
        target_frames=112
    )
    
    val_dataset = CTDataset(
        csv_path=csv_path,
        data_dir=data_dir,
        num_classes=config["num_classes"],
        transform=transform,
        split='test',  # Use test split as validation
        target_frames=112
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"]*8,
        shuffle=False,
        num_workers=12,
        prefetch_factor=4,
        pin_memory=True
    )
    
    # Create model using SlowFast implementation
    model = create_model(
        num_classes=config["num_classes"],
        device=device
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f'Resuming from epoch {start_epoch}')
    
    # Training loop
    early_stopping_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(start_epoch, config["num_epochs"]):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device,
            val_dataloader=val_loader, val_steps=10000
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        logging.info(
            f'Epoch {epoch+1}/{config["num_epochs"]} - '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        )
        
        # Log learning rate
        wandb.log({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    wandb.finish()

if __name__ == '__main__':
    main()