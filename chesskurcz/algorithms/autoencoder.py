import torch
from torch import nn
import numpy as np

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

    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=1000, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.max_game_len = max_game_len
        self.dict_dim = dict_dim
        self.batch_size = batch_size

        self.softmax = nn.Softmax(dim=1)

        self.model = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=max_game_len//8),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//8, out_features=max_game_len//4),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//4, out_features=max_game_len//2),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//2, out_features=max_game_len), 
            nn.ReLU(),   
            nn.Linear(in_features=max_game_len, out_features=dict_dim * max_game_len)
        )

    def forward(self, x):

        x = self.model(x)
        x = torch.reshape(x, (self.batch_size, self.max_game_len, self.dict_dim))

        return self.softmax(x)

class Encoder(nn.Module):

    def __init__(self, max_game_len=200, batch_size=1024, dict_dim=64 * 63, emb_dim=10, channels=4, latent_size=6) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=max_game_len * dict_dim, out_features=max_game_len),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len, out_features=max_game_len//2),
            nn.ReLU(),            
            nn.Linear(in_features=max_game_len//2, out_features=max_game_len//4),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//4, out_features=max_game_len//8),
            nn.ReLU(),
            nn.Linear(in_features=max_game_len//8, out_features=latent_size), 
            nn.ReLU()       
        )

    def forward(self, x):

        return self.model(x.type(torch.float32))
        
class AutoEncoder(nn.Module):
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
        
def main():

    # generate random set
    seq_len = 200
    dict_dim = 1000

    BATCH_SIZE = 32

    N = BATCH_SIZE * 10
    
    seq_origin = torch.randint(0, dict_dim, (N, seq_len))
    data = torch.reshape(nn.functional.one_hot(torch.flatten(seq_origin), dict_dim), (N, seq_len, -1))

    dataset = CustomDataset(data, seq_origin)

    train_loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE, shuffle=True)

    model = AutoEncoder(seq_len, BATCH_SIZE, dict_dim, latent_size=10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

    num_epochs = 200

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, nn.functional.one_hot(labels, dict_dim).type(torch.float32))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct_predictions += (torch.argmax(outputs.data, dim=2) == labels).sum().item()
            total_predictions += torch.flatten(labels).shape[0]
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == "__main__":

    main()