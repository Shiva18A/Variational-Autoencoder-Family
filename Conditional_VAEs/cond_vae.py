import torch
from torch import nn

class VAEEncoder(nn.Module):
    def __init__(self):
        super(VAEEncoder, self).__init__()
        self.num_classes = 10
        self.latent_dim = 50
        self.conv1 = nn.Conv2d(1 + self.num_classes , 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.z_mean = nn.Linear(64*7*7, self.latent_dim)
        self.z_log = nn.Linear(64*7*7, self.latent_dim)
        self.relu = nn.ReLU()
    

    def forward(self, input, labels):
        bs = input.shape[0]
        
        onehot = torch.zeros(labels.size()[0], self.num_classes).to(input.device)
        onehot = onehot.scatter_(1, labels.view(-1, 1), 1)
        onehot = onehot.view(-1, self.num_classes, 1, 1)

        ones = torch.ones((bs, self.num_classes, input.shape[2], input.shape[3]), dtype=input.dtype).to(input.device)
        ones = ones * onehot 
        
        x = torch.cat((input, ones), dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        z_mean = self.z_mean(x)
        z_log = self.z_log(x)

        eps = torch.randn(bs, self.latent_dim, device=input.device)
        z_val = z_mean + torch.exp(z_log / 2) * eps
        return z_mean, z_log, z_val
    


class VAEDecoder(nn.Module):
    def __init__(self):
        super(VAEDecoder, self).__init__()
        self.num_classes = 10
        self.latent_dim = 50
        self.fc1 = nn.Linear(self.latent_dim + self.num_classes, 64*7*7)
        self.reshape = nn.Unflatten(1, (64, 7, 7))
        self.conv1 = nn.ConvTranspose2d(64, 64, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input, labels):
        
        onehot = torch.zeros(labels.size()[0], self.num_classes).to(input.device)
        onehot = onehot.scatter_(1, labels.view(-1, 1), 1)
        
        input = torch.cat((input, onehot), dim=1)
        
        x = self.relu(self.fc1(input))
        x = self.reshape(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        decoded = self.sigmoid(self.conv3(x))
        return decoded
    


class VAEAutoEncoder(nn.Module):
        
    def __init__(self):
        super(VAEAutoEncoder, self).__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def forward(self, input, labels):
        z_mean, z_log, z_val = self.encoder(input, labels)
        decoded = self.decoder(z_val, labels)
        return decoded, z_mean, z_log, z_val
    
    
