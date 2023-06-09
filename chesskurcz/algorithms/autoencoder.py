import torch
from torch import nn
import numpy as np

from chesskurcz.algorithms.autoencoder_utils import default_uci_move_repr

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class Decoder(nn.Module):

    def __init__(self, output_size=(5, 100, 8), batch_size=1024, channels=10, latent_size=6) -> None:
        super().__init__()

        self.max_game_len = output_size[1]
        self.emb_dim = output_size[2]
        self.batch_size = batch_size

        self.fully_layers = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=96),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1, channels, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, (2, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 5, (2,8))
        )

    def forward(self, x):

        x = self.fully_layers(x)
        x = torch.reshape(x, (self.batch_size, 1, 96, 1))
        x = self.conv_layers(x)
        x = torch.reshape(x, (self.batch_size, 1, self.max_game_len, self.emb_dim))

        return x

class Encoder(nn.Module):

    def __init__(self, input_size=(5, 100, 8), kernel_channels=16, latent_size=6) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=kernel_channels, kernel_size=(2,8)),
            nn.ReLU(),
            nn.Conv2d(kernel_channels, kernel_channels, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(kernel_channels, 1, (3, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=input_size[1] - 4, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),            
            nn.Linear(in_features=25, out_features=latent_size),
            nn.ReLU() 
        )

    def forward(self, x):

        return self.model(x.type(torch.float32))
        
class AutoEncoder(nn.Module):
    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.encoder = Encoder(
            (5, 100, 8), channels, latent_size
        )
        self.decoder = Decoder(
            (5, 100, 8), batch_size, channels, latent_size
        )

    def forward(self, x):

        latent = self.encoder(x)

        return self.decoder(latent)
        
def main():

    # generate random set
    seq_len = 200
    dict_dim = 100

    BATCH_SIZE = 16

    N = BATCH_SIZE * 40
    
    seq_origin = torch.randint(0, dict_dim, (N, seq_len))
    data = torch.reshape(nn.functional.one_hot(torch.flatten(seq_origin), dict_dim), (N, seq_len, -1))

    # print(seq_origin)
    # print(data)

    dataset = CustomDataset(data, seq_origin)

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