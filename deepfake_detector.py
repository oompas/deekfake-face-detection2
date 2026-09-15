import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay


# Config
train_real_folder = "data/rvf10k/train/real"
train_fake_folder = "data/rvf10k/train/fake"
test_real_folder = "data/rvf10k/valid/real"
test_fake_folder = "data/rvf10k/valid/fake"
seed = 42
batch_size = 32
num_epochs = 10


# Dataset


class FaceDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path


# Loading data


def load_image_paths(real_folder, fake_folder):
    # real = 1, fake = 0
    all_images = []

    for filename in os.listdir(real_folder):
        if filename.endswith(".jpg"):
            all_images.append((os.path.join(real_folder, filename), 1))

    for filename in os.listdir(fake_folder):
        if filename.endswith(".jpg"):
            all_images.append((os.path.join(fake_folder, filename), 0))

    num_real, num_fake = 0, 0
    for path, label in all_images:
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    print(f"Real: {num_real}, Fake: {num_fake}, Total: {len(all_images)}")
    return all_images


def split_data(all_images, train_ratio=0.8):
    np.random.seed(seed)
    np.random.shuffle(all_images)

    n = len(all_images)
    train_end = int(n * train_ratio)

    train = all_images[:train_end]
    val = all_images[train_end:]

    print(f"Train: {len(train)}, Val: {len(val)}")
    return train, val


def show_samples(images, n=8):
    for i in range(n):
        path, label = images[i]
        print(path)
        img = Image.open(path)
        plt.figure()
        plt.imshow(img)
        plt.title("REAL" if label == 1 else "FAKE")
        plt.show()


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


# Model


def build_model():
    model = models.resnet18(weights='IMAGENET1K_V1')

    # freeze entire network except the final layer
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 1))

    return model


# Training


def train(model, train_loader, val_loader, device):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_pred = 0
        total_pred = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_pred += (preds == labels).sum().item()
            total_pred += labels.size(0)

        train_loss = total_loss / total_pred
        train_acc = correct_pred / total_pred

        val_loss, val_acc = evaluate(model, val_loader, loss_function, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch + 1}/{num_epochs}  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }
    return history


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for images, labels, paths in loader:
            images, labels = images.to(device), labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = loss_function(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_pred += (preds == labels).sum().item()
            total_pred += labels.size(0)
    return total_loss / total_pred, correct_pred / total_pred


def get_predictions(model, test_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for p in probs:
                all_probs.append(p)
            for l in labels.numpy():
                all_labels.append(l)
            for path in paths:
                all_paths.append(path)

    return np.array(all_probs), np.array(all_labels), all_paths


def plot_training_curves(history):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.show()


def plot_results(all_labels, all_probs, all_preds):
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"]).plot()
    plt.title("Confusion Matrix")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load and split training data
    all_images = load_image_paths(train_real_folder, train_fake_folder)
    train_data, val_data = split_data(all_images)
    test_data = load_image_paths(test_real_folder, test_fake_folder)

    #show_samples(train_data)

    train_transform, eval_transform = get_transforms()

    train_dataset = FaceDataset(train_data, transform=train_transform)
    val_dataset = FaceDataset(val_data, transform=eval_transform)
    test_dataset = FaceDataset(test_data, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    model = build_model().to(device)
    history = train(model, train_loader, val_loader, device)
    plot_training_curves(history)

    # Test
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    all_probs, all_labels, all_paths = get_predictions(model, test_loader, device)
    all_preds = (all_probs > 0.5).astype(int)

    print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"]))
    plot_results(all_labels, all_probs, all_preds)


if __name__ == '__main__':
    main()