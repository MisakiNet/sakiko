import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import RewardClassifier as Net

device = torch.device('cuda')
torch.set_default_device(device)


# Define a custom dataset class for our data
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.imgs = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                self.imgs.append((os.path.join(cls_dir, img), cls))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, cls = self.imgs[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label


# Set up data transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Create data loaders for training and testing
train_dataset = ImageDataset('train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
test_dataset = ImageDataset('test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, generator=torch.Generator(device=device))

# Initialize the model, loss function, and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)


def train():
    min_loss = float('inf')
    try:
        os.mkdir('checkpoint')
    except FileExistsError:
        pass

    # Train the model
    for epoch in range(1000):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if loss.item() < min_loss:
                min_loss = loss.item()
                torch.save(model.state_dict(), f'checkpoint/model_{min_loss:.6f}.pt')
                print(f'Saved model with loss {min_loss:.6f}')
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}')

    print('Finished Training')


def test(use_model):
    # Test the model
    model.eval()
    test_loss = 0
    correct = 0

    # restore best model
    print(f'Restoring model {use_model}')
    model.load_state_dict(torch.load(use_model))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    # print(f'Test Loss: {test_loss / len(test_loader)}')
    # print(f'Test Accuracy: {accuracy:.4f}%')
    return test_loss / len(test_loader), accuracy


if __name__ == '__main__':
    # train()
    # test('checkpoint/model_0.000000.pt')
    models = os.listdir('checkpoint')
    for i in range(10):
        print(f'Model {models[i]}: {test(f"checkpoint/{models[i]}")}')
