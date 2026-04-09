# modules/cnn_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# ──────────────────────────────────────────
# PART A: Model Definition
# ──────────────────────────────────────────

def create_model():
    """
    Creates a ResNet18 model modified for binary classification.
    
    ResNet18 = A deep neural network pretrained on ImageNet
               (1.2 million images, 1000 categories)
    
    We take this pretrained model and replace the last layer
    to output only 2 values: [Real, Forged]
    
    This technique is called TRANSFER LEARNING.
    """
    
    # Load ResNet18 with pretrained ImageNet weights
    # pretrained=True means it already knows edges, textures, shapes
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers — we don't want to retrain basic feature detectors
    # We only want to train the last few layers on our specific data
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    
    # Unfreeze last 2 layers (layer4 and fc) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace final fully connected layer
    # Original: 512 → 1000 (ImageNet has 1000 classes)
    # Ours:     512 → 2    (Real or Forged)
    num_features = model.fc.in_features  # = 512
    model.fc = nn.Sequential(
        nn.Dropout(0.5),           # Dropout prevents overfitting
        nn.Linear(num_features, 256),
        nn.ReLU(),                 # Activation function
        nn.Linear(256, 2)          # Final: 2 outputs
    )
    
    return model


# ──────────────────────────────────────────
# PART B: Image Preprocessing
# ──────────────────────────────────────────

def get_transforms():
    """
    Defines image transformations for model input.
    Neural networks need fixed-size, normalized images.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),    # ResNet expects 224x224
        transforms.ToTensor(),             # Convert PIL Image → PyTorch Tensor
        transforms.Normalize(              # Normalize using ImageNet mean/std
            mean=[0.485, 0.456, 0.406],   # These values are standard for ResNet
            std=[0.229, 0.224, 0.225]
        )
    ])


# ──────────────────────────────────────────
# PART C: Training
# ──────────────────────────────────────────

def train_model(dataset_path, save_path='model/forgery_model.pth', epochs=10):
    """
    Trains the CNN on your dataset.
    
    dataset_path should have this structure:
    dataset/
      real/      ← Put real document images here
      forged/    ← Put forged document images here
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, random_split
    
    print("🔄 Loading dataset...")
    
    # ImageFolder automatically assigns labels based on folder names
    # dataset/real   → label 1
    # dataset/forged → label 0
    full_dataset = ImageFolder(
        root=dataset_path,
        transform=get_transforms()
    )
    
    print(f"✅ Found {len(full_dataset)} images")
    print(f"   Classes: {full_dataset.classes}")
    
    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # DataLoader: loads images in batches during training
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Setup model, loss function, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    model = create_model().to(device)
    
    # CrossEntropyLoss: standard loss for classification problems
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer: adjusts learning rate automatically
    # Only optimize parameters with requires_grad=True (unfrozen layers)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    
    # Learning rate scheduler: reduce LR if accuracy stops improving
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_acc = 0
    
    # ── Training Loop ──
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()       # Clear previous gradients
            outputs = model(images)     # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()             # Backpropagation
            optimizer.step()            # Update weights
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        
        # ── Validation ──
        model.eval()  # Set model to evaluation mode
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # No gradient calculation needed for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Model saved! Best Val Acc: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    print("✅ Training complete!")
    return model


# ──────────────────────────────────────────
# PART D: Prediction
# ──────────────────────────────────────────

def predict_image(image_path, model_path='model/forgery_model.pth'):
    """
    Predicts whether a single image is Real or Forged.
    Returns: (label, confidence_score)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model architecture
    model = create_model().to(device)
    
    # Load saved weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("⚠️ No trained model found. Using untrained model.")
    
    model.eval()  # Evaluation mode: disables dropout
    
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)
    # unsqueeze(0) adds batch dimension: [C, H, W] → [1, C, H, W]
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Softmax converts raw scores to probabilities (sum to 1)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, 1)
        
    # Class mapping (depends on ImageFolder alphabetical order)
    # 'forged' comes before 'real' alphabetically
    class_names = ['forged', 'real']
    label = class_names[predicted_class.item()]
    score = confidence.item() * 100
    
    return label, round(score, 2)