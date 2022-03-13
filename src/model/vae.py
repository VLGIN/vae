import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()

        # create encoder with input_dim and hidden_dim as dim of z
        self.encoder = Encoder(input_dim, hidden_dim)

        # create decoder with input_dim is dim of z and output_dim is image dim
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, x):
        (mu, sigma) = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        output = self.decoder(z)
        loss = self.loss(mu, sigma, x, output)
        return {"loss": loss, "output": output}

    def reparameterize(self, mu, sigma):
        """
        Sample z by reparameterization trick
        """
        # temp = torch.exp(0.5 * sigma)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma       

    def loss(self, mu, sigma, x, output):
        """
        Calculate loss for VAE
        """
        kld = torch.mean(0.5*torch.sum(1 + sigma - mu**2 - torch.exp(sigma), dim=1), dim=0)

        loss = kld + torch.nn.functional.mse_loss(output, x)
        return loss

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        image_dim = 1
        for i in input_dim:
            image_dim *= i
        
        self.flatten = nn.Flatten(start_dim=1)
        self.mu = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.sigma = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return (mu, sigma)

class Decoder(nn.Module):
    def __init__(self, input_dim, image_dim):
        super(Decoder, self).__init__()
        self.image_dim = [-1] + list(image_dim)
        output_dim = 1
        for i in image_dim:
            output_dim *= i
        self.h = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        output = self.h(x)
        return output.view(self.image_dim)