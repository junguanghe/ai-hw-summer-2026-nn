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
PATCH_SIZE = 4       # 28x28 image divided into 4x4 patches -> (7x7 = 49 patches)
NUM_PATCHES = (28 // PATCH_SIZE) ** 2  # 49
PATCH_DIM = PATCH_SIZE * PATCH_SIZE   # 16
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
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

# 3. Model Definition: Vision Transformer Encoder
class ViTEncoder(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, num_patches=NUM_PATCHES, patch_dim=PATCH_DIM,
                 d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, num_classes=10):
        super(ViTEncoder, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear projection of flattened patches
        self.patch_embedding = nn.Linear(patch_dim, d_model)

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (B, 1, 28, 28)
        B = x.shape[0]

        # Extract non-overlapping patches and flatten them: (B, 49, 16)
        # Using unfold: 28x28 -> (B, 1, 7, 7, 4, 4) -> reshape to (B, 49, 16)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)  # (B, 49, 16)

        # Embed patches: (B, 49, d_model)
        x_embed = self.patch_embedding(patches)

        # Expand CLS token for batch: (B, 1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS token to patch embeddings: (B, 50, d_model)
        x_seq = torch.cat((cls_tokens, x_embed), dim=1)

        # Add positional embedding
        x_seq = x_seq + self.pos_embedding

        # Pass through Transformer Encoder
        out = self.transformer_encoder(x_seq)

        # Take [CLS] token output (first element in sequence): (B, d_model)
        cls_out = out[:, 0, :]

        # Classification logits
        logits = self.fc_head(cls_out)
        return logits

# 4. Initialize Model, Loss Function, and Optimizer
model = ViTEncoder().to(device)
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
    plt.title('Transformer Encoder: Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'o-', label='Train Accuracy')
    plt.plot(epochs, history["test_acc"], 'o-', label='Test Accuracy')
    plt.title('Transformer Encoder: Training and Testing Accuracy (%)')
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
    print("Starting training Transformer (Encoder) on MNIST...\n")

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    best_accuracy = 0.0
    model_save_path = os.path.join(SAVE_DIR, "transformer_mnist_best.pth")
    metrics_save_path = os.path.join(SAVE_DIR, "transformer_metrics.json")
    plot_save_path = os.path.join(SAVE_DIR, "transformer_learning_curves.png")

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
