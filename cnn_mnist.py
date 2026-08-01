import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Hyperparameters & Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SAVE_DIR = "./results"

os.makedirs(SAVE_DIR, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data Loading & Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST standard mean and std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Definition: Convolutional Neural Network (CNN)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # First Conv Block: (1, 28, 28) -> (32, 14, 14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second Conv Block: (32, 14, 14) -> (64, 7, 7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully Connected Block: 64 * 7 * 7 = 3136 -> 128 -> 10
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 4. Initialize Model, Loss Function, and Optimizer
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training Loop
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

# 6. Evaluation Loop
def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    total_loss = test_loss / total
    accuracy = 100.0 * correct / total
    print(f"--> Test Loss: {total_loss:.4f} | Test Accuracy: {accuracy:.2f}%\n")
    return total_loss, accuracy

# 7. Plot and Save Learning Curves
def plot_metrics(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'o-', label='Train Loss')
    plt.plot(epochs, history["test_loss"], 'o-', label='Test Loss')
    plt.title('CNN: Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'o-', label='Train Accuracy')
    plt.plot(epochs, history["test_acc"], 'o-', label='Test Accuracy')
    plt.title('CNN: Training and Testing Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to: {save_path}")

if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Starting training CNN on MNIST...\n")

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    best_accuracy = 0.0
    model_save_path = os.path.join(SAVE_DIR, "cnn_mnist_best.pth")
    metrics_save_path = os.path.join(SAVE_DIR, "cnn_metrics.json")
    plot_save_path = os.path.join(SAVE_DIR, "cnn_learning_curves.png")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer, epoch)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        # Save model checkpoint if test accuracy improves
        if te_acc > best_accuracy:
            best_accuracy = te_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model checkpoint with accuracy: {best_accuracy:.2f}% to {model_save_path}")

    # Save metrics JSON
    with open(metrics_save_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved metrics log to: {metrics_save_path}")

    # Plot and save curves
    plot_metrics(history, plot_save_path)

    print(f"\nTraining completed! Best Test Accuracy: {best_accuracy:.2f}%")
