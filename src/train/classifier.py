"""
Train a card attribute classifier on the existing labeled images.

Uses MobileNetV3-Small for iPhone compatibility.
Multi-head output: predicts all 4 attributes simultaneously.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import numpy as np
from tqdm import tqdm

# === Config ===

DATA_DIR = Path(__file__).parent.parent.parent / "training_images"
SYNTHETIC_DATA_DIR = Path(__file__).parent.parent.parent / "training_images_synthetic"
WEIGHTS_DIR = Path(__file__).parent.parent.parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# Attribute mappings (folder names → indices)
NUMBER_MAP = {"one": 0, "two": 1, "three": 2}
COLOR_MAP = {"red": 0, "green": 1, "blue": 2}  # blue = purple in standard Set
SHAPE_MAP = {"diamond": 0, "oval": 1, "squiggle": 2}
FILL_MAP = {"empty": 0, "full": 1, "partial": 2}  # partial = striped

# Reverse mappings for inference
NUMBER_NAMES = ["one", "two", "three"]
COLOR_NAMES = ["red", "green", "blue"]
SHAPE_NAMES = ["diamond", "oval", "squiggle"]
FILL_NAMES = ["empty", "full", "partial"]


# === Dataset ===

class SetCardDataset(Dataset):
    """Dataset of labeled Set card images."""

    def __init__(self, data_dirs, transform=None):
        if isinstance(data_dirs, Path):
            data_dirs = [data_dirs]
        self.transform = transform
        self.samples: List[Tuple[Path, Dict[str, int]]] = []

        # Walk the directory structure to find all images
        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            count_before = len(self.samples)
            for number in NUMBER_MAP:
                for color in COLOR_MAP:
                    for shape in SHAPE_MAP:
                        for fill in FILL_MAP:
                            folder = data_dir / number / color / shape / fill
                            if folder.exists():
                                for img_path in folder.glob("*.png"):
                                    labels = {
                                        "number": NUMBER_MAP[number],
                                        "color": COLOR_MAP[color],
                                        "shape": SHAPE_MAP[shape],
                                        "fill": FILL_MAP[fill],
                                    }
                                    self.samples.append((img_path, labels))
            print(f"Loaded {len(self.samples) - count_before} samples from {data_dir}")

        print(f"Total: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Stack labels into tensor
        label_tensor = torch.tensor([
            labels["number"],
            labels["color"],
            labels["shape"],
            labels["fill"],
        ], dtype=torch.long)
        
        return image, label_tensor
    
    def get_raw(self, idx):
        """Get raw PIL image and labels (no transform)."""
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label_tensor = torch.tensor([
            labels["number"],
            labels["color"],
            labels["shape"],
            labels["fill"],
        ], dtype=torch.long)
        return image, label_tensor


# === Model ===

class SetCardClassifier(nn.Module):
    """
    Multi-head classifier for Set card attributes.
    
    Uses MobileNetV3-Small backbone (good for mobile deployment).
    Four output heads, one per attribute.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Get the feature dimension from the classifier
        in_features = self.backbone.classifier[0].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add our multi-head classifier
        self.heads = nn.ModuleDict({
            "number": nn.Linear(in_features, 3),
            "color": nn.Linear(in_features, 3),
            "shape": nn.Linear(in_features, 3),
            "fill": nn.Linear(in_features, 3),
        })
    
    def forward(self, x):
        features = self.backbone(x)
        return {
            "number": self.heads["number"](features),
            "color": self.heads["color"](features),
            "shape": self.heads["shape"](features),
            "fill": self.heads["fill"](features),
        }


# === Training ===

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = {k: 0 for k in ["number", "color", "shape", "fill"]}
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss for each head (2x weight on fill to penalize fill mistakes)
        loss = 0
        fill_weight = 2.0
        for i, key in enumerate(["number", "color", "shape", "fill"]):
            head_loss = criterion(outputs[key], labels[:, i])
            loss += fill_weight * head_loss if key == "fill" else head_loss
            preds = outputs[key].argmax(dim=1)
            correct[key] += (preds == labels[:, i]).sum().item()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracies = {k: v / total for k, v in correct.items()}
    return avg_loss, accuracies


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = {k: 0 for k in ["number", "color", "shape", "fill"]}
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            loss = 0
            for i, key in enumerate(["number", "color", "shape", "fill"]):
                loss += criterion(outputs[key], labels[:, i])
                preds = outputs[key].argmax(dim=1)
                correct[key] += (preds == labels[:, i]).sum().item()
            
            total_loss += loss.item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracies = {k: v / total for k, v in correct.items()}
    return avg_loss, accuracies


def main():
    # === Hyperparameters ===
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.10
    IMG_SIZE = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # === Data transforms ===
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),  # Simulate imperfect detector crops
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),  # Cards can be any orientation
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),  # Perspective warp from detection
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.05),  # Force model to not rely solely on color for fill
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # ~30% effective via random sigma
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # === Load dataset (clean + synthetic crops) ===
    data_dirs = [DATA_DIR]
    if SYNTHETIC_DATA_DIR.exists():
        data_dirs.append(SYNTHETIC_DATA_DIR)
    full_dataset = SetCardDataset(data_dirs, transform=None)  # No transform yet
    
    # Split into train/val/test
    total = len(full_dataset)
    test_size = int(total * TEST_SPLIT)
    val_size = int(total * VAL_SPLIT)
    train_size = total - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Wrap with transform (can't change transform on Subset, so we wrap)
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    test_dataset = TransformDataset(test_dataset, val_transform)
    
    # Use num_workers=0 on macOS to avoid shared memory issues
    import platform
    num_workers = 0 if platform.system() == "Darwin" else 4
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    # === Model ===
    model = SetCardClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # === Training loop ===
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        # Average accuracy across all heads
        avg_train_acc = sum(train_acc.values()) / 4
        avg_val_acc = sum(val_acc.values()) / 4
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {avg_val_acc:.4f}")
        print(f"  Val per-head: num={val_acc['number']:.3f} col={val_acc['color']:.3f} "
              f"shp={val_acc['shape']:.3f} fil={val_acc['fill']:.3f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, WEIGHTS_DIR / "classifier_best.pt")
            print(f"  Saved new best model (val_acc={avg_val_acc:.4f})")
    
    # === Final evaluation on test set ===
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(WEIGHTS_DIR / "classifier_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    avg_test_acc = sum(test_acc.values()) / 4
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy (avg): {avg_test_acc:.4f}")
    print(f"  Number: {test_acc['number']:.4f}")
    print(f"  Color:  {test_acc['color']:.4f}")
    print(f"  Shape:  {test_acc['shape']:.4f}")
    print(f"  Fill:   {test_acc['fill']:.4f}")
    
    # Save final results
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "avg_test_accuracy": avg_test_acc,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }
    with open(WEIGHTS_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to {WEIGHTS_DIR / 'classifier_best.pt'}")
    print(f"Results saved to {WEIGHTS_DIR / 'training_results.json'}")


if __name__ == "__main__":
    main()
