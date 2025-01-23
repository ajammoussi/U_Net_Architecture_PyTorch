import os
import torch
import yaml
from src.u_net import UNetModel

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize model
def initialize_model(config, device):
    model = UNetModel(config)
    model = model.model.to(device)
    return model

# Save model
def save_model(model, save_path):
    torch.save(model.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Load model
def load_model(model, load_path):
    if os.path.exists(load_path):
        model.model.load_state_dict(torch.load(load_path, weights_only=True))
        print(f"Model loaded from {load_path}")
        return True
    else:
        print(f"No model found at {load_path}")
        return False

# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # Initialize model
    unet_model = initialize_model(config, device)

    # Define model save path
    model_save_path = os.path.join(config['train']['working_dir'], "model.pth")

    # Check if model exists and load it
    model_exists = load_model(unet_model, model_save_path)

    # Train the model if it doesn't exist
    if not model_exists:
        print("Starting training...")
        unet_model.train_UNet()
        save_model(unet_model, model_save_path)
    else:
        print("Skipping training as the model already exists.")

    # Test the model
    print("Starting testing...")
    unet_model.test_UNet()

if __name__ == "__main__":
    main()