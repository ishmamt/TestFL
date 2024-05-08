import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4) # Flatten the output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    

def train(model, train_loader, optimizer, epochs, device):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = CrossEntropyLoss()
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {len(epochs)}:\n")
        loop = tqdm(train_loader)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(f"Loss: {loss.item():.3f}")
            
    print(f"Training finished.\n")


def test(model, test_loader, device):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = CrossEntropyLoss()
    correct, loss = 0, 0.0
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        loop = tqdm(test_loader)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(f"Loss: {loss.item():.3f}")
    
    accuracy = correct / len(test_loader.dataset)
    
    return loss, accuracy