import torch
import torch.nn as nn
import numpy as np
import random
import os


def set_seed(seed):
    """
    Set random seeds for reproducibility across different devices and modules.

    Args:
        seed (int): The random seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConceptAutoEncoder(nn.Module):
    """
    AutoEncoder network designed for learning the latent feature manifold
    and extracting significant concept features.

    Used in both the manifold learning stage (discriminator.py) and
    the graph construction stage (graph.py).
    """

    def __init__(self, input_dim, encoding_layers):
        super(ConceptAutoEncoder, self).__init__()

        # Encoder Construction
        encoder_layers = []
        in_features = input_dim
        for out_features in encoding_layers:
            encoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_features = out_features
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder Construction
        decoder_layers = []
        for out_features in reversed(encoding_layers[:-1]):
            decoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_features = out_features
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded