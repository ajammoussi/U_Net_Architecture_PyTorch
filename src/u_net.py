import torch
from src.model.model import UNet
from src.model.train import launch_train
from src.test.test import launch_test


class UNetModel:
    device = None

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(config['model']['in_channels'], config['model']['out_channels'], config['model']['init_features']).to(self.device)
        self.config = config

    def train_UNet(self):
        launch_train(self.device, self.model, self.config)

    def test_UNet(self):
        launch_test(self.device, self.model, self.config)
