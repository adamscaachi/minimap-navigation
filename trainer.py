import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

class Trainer:

    def __init__(self, model, criterion, optimiser, device):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device

    def data_split(self, images, angles):
        images_1, images_2, angles_1, angles_2 = train_test_split(images, angles, test_size=0.2, random_state=42)
        return images_1, images_2, angles_1, angles_2

    def loader(self, images, angles):
        images = (np.array(images)[:, :, :, :3].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        images_tensor = torch.tensor(images, dtype=torch.float32)
        angles = np.array(angles)
        x = torch.tensor(np.cos(np.deg2rad(angles)), dtype=torch.float32).view(-1, 1)
        y = torch.tensor(np.sin(np.deg2rad(angles)), dtype=torch.float32).view(-1, 1)
        angles_tensor = torch.cat((x, y), dim=1)
        dataset_tensor = TensorDataset(images_tensor, angles_tensor)
        return DataLoader(dataset_tensor, batch_size=64, shuffle=True)

    def train(self, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimiser.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimiser.step()
                running_loss += loss.item() * inputs.size(0)
            train_losses.append(running_loss / len(train_loader.dataset))
            val_losses.append(self._validate(val_loader))
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_losses[epoch]:.4f}, Validation Loss: {val_losses[epoch]:.4f}")
        self._plot_losses(num_epochs, train_losses, val_losses)

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        return val_loss

    def _plot_losses(self, num_epochs, train_losses, val_losses):
        x = np.linspace(1, num_epochs, num_epochs)
        plt.plot(x[1:], train_losses[1:], 'b-', label='Training')
        plt.plot(x[1:], val_losses[1:], 'r-', label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curves.png', dpi=400, bbox_inches='tight')