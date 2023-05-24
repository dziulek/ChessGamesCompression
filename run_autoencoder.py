import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import os
import numpy as np

from chesskurcz.algorithms.autoencoder import CustomDataset, AutoEncoder
from chesskurcz.algorithms.utils import get_all_possible_uci_moves, get_script_path
from chesskurcz.algorithms.transform import game_from_pgn_to_uci
from chesskurcz.algorithms.autoencoder_utils import default_uci_move_repr

BATCH_SIZE = 16

N = BATCH_SIZE * 40
NULL_TOKEN = '<>'

def main():

    # generate set
    uci_moves_dict = get_all_possible_uci_moves()
    uci_moves_dict[NULL_TOKEN] = len(uci_moves_dict)
    NUMBER_OF_TOKENS = len(uci_moves_dict)
    print('NUMBER OF TOKENS: ', NUMBER_OF_TOKENS)

    data_file_path = os.path.join(get_script_path(), 'data/filtered_test_file.pgn')

    game_str = None
    with open(data_file_path, 'r') as f:
        game_str = f.readlines()

    games_uci = [game_from_pgn_to_uci(l) for l in game_str]

    # fix length
    for i, g in enumerate(games_uci):
        if len(g) < 100:
            while len(games_uci[i]) < 100:
                games_uci[i].append('')
        
        games_uci[i] = games_uci[i][:100]

    games_mapped = np.zeros((len(games_uci), 5, 100, 8))
    # map uci to numbers
    for i in range(len(games_uci)):
        for j in range(100):
            games_mapped[i][:][j] = np.concatenate((default_uci_move_repr(games_uci[i][j])))
    
    games_tensor = torch.tensor(games_mapped)
    data = torch.reshape(games_tensor, (len(games_mapped),1,  100, -1))

    # print(seq_origin)
    # print(data)

    dataset = CustomDataset(data, data)

    train_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

    model = AutoEncoder(data.shape[1], BATCH_SIZE, NUMBER_OF_TOKENS, latent_size=20, channels=16)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

    num_epochs = 1000

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in train_loader:

            outputs = model(images)
            loss = criterion(outputs, labels.type(torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')   

if __name__ == "__main__":

    main()