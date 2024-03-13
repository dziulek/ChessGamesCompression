import torch
from torch import nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os, pathlib

import chess.pgn

from chesskurcz.algorithms.util.autoencoder_utils import default_uci_move_repr, DEF_REPR_T_SIZE
from chesskurcz.algorithms import encoder
from chesskurcz.algorithms.transform import game_from_pgn_to_uci


class ChessGameDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        game = self.data[index]
        target = self.targets[index]
        
        if self.transform:
            game = self.transform(game)
        
        return game, target

class DecoderNN(nn.Module):

    def __init__(self, 
                 output_size: Tuple, 
                 batch_size: int, 
                 dense: List[int],
                 conv_levels: int,
                 channels: int | List[int],
                 kernels: int | Tuple | List[Tuple | int],
                 latent_size=6) -> None:
        super().__init__()

        self.max_game_len = output_size[1]
        self.emb_dim = output_size[2]
        self.batch_size = batch_size

        self.fully_layers = None
        if len(dense) > 0:

            self.fully_layers = nn.Sequential()
            dense.insert(0, latent_size)
            for i in range(dense[:-1]):
                self.fully_layers.add_module(
                    name=f"DENSE:{dense[i]}:{dense[i + 1]}",
                    module=nn.Linear(in_features=dense[i], out_features=dense[i + 1])
                )
                self.fully_layers.add_module(nn.ReLU())
        
        self.conv_layers = None
        if conv_levels:
            if isinstance(channels, int):
                channels = [channels for i in range(conv_levels)]
            if isinstance(kernels, int):
                kernels = (kernels, kernels)
            if isinstance(kernels, Tuple):
                kernels = [kernels for i in range(conv_levels)]

        self.conv_layers = nn.Sequential()
        channels.insert(0, 1)
        for i in range(channels[:-1]):
            self.conv_layers.add_module(
                name=f"TRANS_CONV:{channels[i]}:{channels[i + 1]}:({kernels[i][0], kernels[i][1]})",
                module=nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    stride=1
                )
            )

    def forward(self, x):
        
        if self.fully_layers is not None:
            x = self.fully_layers(x)
        x = torch.reshape(x, (self.batch_size, 1, -1, 1))

        if self.conv_layers is not None:
            x = self.conv_layers(x)
        x = torch.reshape(x, (self.batch_size, 1, self.max_game_len, self.emb_dim))

        return x

class EncoderNN(nn.Module):

    def __init__(self, 
                 input_size: Tuple, 
                 batch_size: int, 
                 dense: List[int],
                 conv_levels: int,
                 channels: int | List[int],
                 kernels: int | Tuple | List[Tuple | int]) -> None:
        super().__init__()

        self.max_game_len = input_size[1]
        self.emb_dim = input_size[2]
        self.batch_size = batch_size

        self.fully_layers = None
        if len(dense) > 0:

            self.fully_layers = nn.Sequential()
            dense.insert(0, latent_size)
            for i in range(dense[:-1]):
                self.fully_layers.add_module(
                    name=f"DENSE:{dense[i]}:{dense[i + 1]}",
                    module=nn.Linear(in_features=dense[i], out_features=dense[i + 1])
                )
                self.fully_layers.add_module(nn.ReLU())
        
        self.conv_layers = None
        if conv_levels:
            if isinstance(channels, int):
                channels = [channels for _ in range(conv_levels)]
            if isinstance(kernels, int):
                kernels = (kernels, kernels)
            if isinstance(kernels, Tuple):
                kernels = [kernels for _ in range(conv_levels)]

        self.conv_layers = nn.Sequential()
        channels.insert(0, 1)
        for i in range(channels[:-1]):
            self.conv_layers.add_module(
                name=f"CONV:{channels[i]}:{channels[i + 1]}:({kernels[i][0], kernels[i][1]})",
                module=nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    stride=1
                )
            )

    def forward(self, x):

        if self.conv_layers is not None:
            x = self.conv_layers(x)
        x = nn.Flatten()(x)

        if self.fully_layers is not None:
            x = self.fully_layers(x)
        
        return x
        
class AutoEncoder(nn.Module):
    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.encoder = EncoderNN(
            (5, 100, 8), channels, latent_size
        )
        self.decoder = DecoderNN(
            (5, 100, 8), batch_size, channels, latent_size
        )
        # dummy commment
    def forward(self, x):

        latent = self.encoder(x)

        return self.decoder(latent)

def read_games(path: str) -> List[List[str]]:

    games = []

    with open(path, 'r') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            game_uci = game_from_pgn_to_uci(game)
            games.append(game_uci)

    return games

def main():

    pgn_data_path = os.path.join(pathlib.Path(__file__).absolute().parents[2], 'data', 'test_file.pgn')
    games_data = read_games(pgn_data_path)

    seq_len = max([len(g) for g in games_data])

    for game in games_data:
        game += [''] * (seq_len - len(game))
        for i, move in enumerate(game):
            game[i] = default_uci_move_repr(move)

        game = torch.concatenate(game)

    seq_len = 200
    dict_dim = 100

    BATCH_SIZE = 16

    N = BATCH_SIZE * 40

    enc = encoder.Encoder(alg='apm', par_workers=4, batch_size=10000)
    games = enc.decode()
    
    seq_origin = torch.randint(0, dict_dim, (N, seq_len))
    data = torch.reshape(nn.functional.one_hotK(torch.flatten(seq_origin), dict_dim), (N, seq_len, -1))
    # print(seq_origin)
    # print(data) dummy comment
    dataset = ChessGameDataset(data, seq_origin)
    
    train_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

    model = AutoEncoder(seq_len, BATCH_SIZE, dict_dim, latent_size=10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

    num_epochs = 1000

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in train_loader:

            outputs = model(images)
            outputs = torch.reshape(outputs, (-1, dict_dim))
            one_hot_labels = nn.functional.one_hot(labels, dict_dim).type(torch.float32)
            loss = criterion(outputs, torch.reshape(one_hot_labels, (-1, dict_dim)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_labels = torch.argmax(outputs.data, dim=1)
            
            correct_predictions += (output_labels == labels.flatten()).sum().item()
            total_predictions += torch.flatten(labels).shape[0]
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')           

if __name__ == "__main__":

    main()