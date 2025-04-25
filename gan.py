import torch
import torch.nn as nn

class GAN(nn.Module):  # Should inherit from nn.Module
    def __init__(self, noise_dim, signal_length=784):
        super(GAN, self).__init__()
        self.noise_dim = noise_dim
        self.signal_length = signal_length  # For MNIST: 28x28=784
        
        # Generator (Encoder-Decoder)
        self.generator_encoder = nn.Sequential(
            nn.Conv1d(1, 64, 4, stride=2, padding=1),  # [B, 64, L/2]
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),  # [B, 128, L/4]
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),  # [B, 256, L/8]
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        
        self.generator_decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),  # [B, 128, L/4]
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),  # [B, 64, L/2]
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, 4, stride=2, padding=1),  # [B, 1, L]
            nn.Tanh()
        )

        # Discriminator (1D Convs)
        self.discriminator = nn.Sequential(
            nn.Conv1d(1, 64, 4, stride=2, padding=1),  # [B, 64, L/2]
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),  # [B, 128, L/4]
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),  # [B, 256, L/8]
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )

    def forward_generator(self, x):
        encoded = self.generator_encoder(x)
        decoded = self.generator_decoder(encoded)
        return decoded

    def forward_discriminator(self, x):
        return self.discriminator(x)