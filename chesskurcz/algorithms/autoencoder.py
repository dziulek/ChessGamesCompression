from torch import nn
import numpy as np
import math

import json
import sys, os, io

from torch.nn import Module, Linear, Conv1d, ReLU, Softmax
class Decoder(nn.Module):

    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=max_game_len//8),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//8, out_features=max_game_len//4),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//4, out_features=max_game_len//2),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//2, out_features=max_game_len - 4), 
            nn.ReLU(),   
            nn.BatchNorm2d(1), 
            nn.ConvTranspose2d(in_channels=1, out_channels=channels, kernel_size=(2, emb_dim)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=channels, out_channels=1, kernel_size=(2, emb_dim)),
            nn.ReLU(),                     
        )

    def forward(self, x):

        return self.model(x)

class Encoder(nn.Module):

    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Embedding(num_embeddings=dict_dim, embedding_dim=emb_dim),
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=(2, emb_dim)),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Linear(in_features=max_game_len - 4, out_features=max_game_len//2),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//2, out_features=max_game_len//4),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//4, out_features=max_game_len//8),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//8, out_features=latent_size), 
            nn.ReLU()       
        )

    def forward(self, x):

        return self.model(x)
        
class AutoEncoder(Module):
    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.encoder = Encoder(
            max_game_len, batch_size, dict_dim, emb_dim, channels, latent_size
        )
        self.decoder = Decoder(
            max_game_len, batch_size, dict_dim, emb_dim, channels, latent_size
        )

    def forward(self, x):

        latent = self.encoder(x)

        return self.decoder(latent)
        

