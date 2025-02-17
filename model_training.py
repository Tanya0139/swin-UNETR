import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from monai.networks.nets import UNETR
from monai.transforms import Resize, ScaleIntensity, Compose
import pandas as pd

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directory
DATASET_DIR = r"C:\Users\tanya\OneDrive\Desktop\tantan\combined_work_pjt\BRATS_2020_DATA\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# Load MRI scan
def load_mri_scan(image_path):
    img = nib.load(image_path).get_fdata()
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize intensity
    return img

# Preprocess MRI scan
def preprocess_mri(image, target_shape=(96, 96, 96)):
    transform = Resize(target_shape)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
    return transform(image).squeeze(0).numpy()  # Convert back to NumPy

# Load dataset
def load_brats_dataset(data_dir):
    images, masks = [], []
    subject_folders = sorted(os.listdir(data_dir))
    print(f"Found {len(subject_folders)} subject folders.")
    for subject_dir in subject_folders:
        subject_path = os.path.join(data_dir, subject_dir)
        if os.path.isdir(subject_path):
            flair_path = os.path.join(subject_path, f"{subject_dir}_flair.nii")
            seg_path = os.path.join(subject_path, f"{subject_dir}_seg.nii")
            if os.path.exists(flair_path) and os.path.exists(seg_path):
                print(f"Loading: {flair_path}, {seg_path}")
                img = preprocess_mri(load_mri_scan(flair_path))
                mask = preprocess_mri(load_mri_scan(seg_path))
                images.append(img)
                masks.append(mask)
            else:
                print(f"Missing files in {subject_dir}")
    print(f"Dataset loading complete: {len(images)} images and {len(masks)} masks successfully loaded.")
    images = np.expand_dims(np.array(images), axis=1)  # Add channel dimension
    masks = np.expand_dims(np.array(masks), axis=1)  # Add channel dimension
    return images, masks

# Define dataset class
class MRIDataset(data.Dataset):
    def __init__(self, images, masks, transform=None):
        super().__init__()
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask = self.images[idx], self.masks[idx]
        if self.transform:
            img, mask = self.transform(img), self.transform(mask)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, patience=5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device, dtype=torch.float16), masks.to(device, dtype=torch.float16)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, dtype=torch.float16), masks.to(device, dtype=torch.float16)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = r"C:\Users\tanya\OneDrive\Desktop\tantan\combined_work_pjt\saved_model_2020\unetr_finetuned.pth"
            torch.save(model.state_dict(), save_path)
            print("Model improved and saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Ensures correct multiprocessing on Windows

    images, masks = load_brats_dataset(DATASET_DIR)

    # Create dataset
    transform = Compose([ScaleIntensity()])
    dataset = MRIDataset(images, masks, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders with num_workers=0 for Windows stability
    train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=4, num_workers=0, pin_memory=True)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Load UNETR model
    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(96, 96, 96),
        feature_size=16,
    ).to(device)

    # Train the model
    train_model(model, train_loader, val_loader)
