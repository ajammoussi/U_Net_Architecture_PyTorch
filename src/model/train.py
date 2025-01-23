import torch
import torch.nn as nn
from src.data.dataloaders import create_train_dataloader

# Training the model
def train_model(model, criterion, optimizer, train_loader, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            assert inputs.dtype == torch.float32, f"Expected inputs to be float32, but got {inputs.dtype}"
            assert targets.dtype == torch.long, f"Expected targets to be long, but got {targets.dtype}"
            assert inputs.shape[1] == 3, f"Expected inputs to have 3 channels, but got {inputs.shape[1]}"
            assert targets.shape[1] == 1, f"Expected targets to have 1 channel, but got {targets.shape[1]}"
            inputs, targets = inputs.to(device), targets.to(device)

            targets = (targets > 1.0).float() # Ensure targets are floats and binary

            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == targets.shape, f"Expected outputs shape {targets.shape}, but got {outputs.shape}"
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def launch_train(device, model, config):
    # Define paths and parameters
    working_dir = config['train']['working_dir']
    train_batch_size = config['train']['train_batch_size']
    epochs = config['train']['epochs']

    # Create dataloaders
    train_loader = create_train_dataloader(working_dir, train_batch_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    train_model(model, criterion, optimizer, train_loader, device, epochs)
