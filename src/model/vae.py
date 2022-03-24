import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, image_size, num_cnn, latent_dim):
        super(VAE, self).__init__()

        # create encoder with input_dim and hidden_dim as dim of z
        self.encoder = Encoder(image_size, num_cnn, latent_dim)

        # create decoder with input_dim is dim of z and output_dim is image dim
        self.decoder = Decoder(image_size, num_cnn, latent_dim)

    def forward(self, x):
        (mu, sigma) = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        output = self.decoder(z)
        loss = self.loss(mu, sigma, x, output)
        return {"loss": loss, "output": output}

    def reparameterize(self, mu, log_var):
        """
        Sample z by reparameterization trick
        """
        # temp = torch.exp(0.5 * sigma)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, mu, log_var, x, output):
        """
        Calculate loss for VAE
        """
        kld = torch.mean(-0.5*torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=1), dim=0)

        loss = kld + torch.nn.functional.mse_loss(output, x)
        return loss

class Encoder(nn.Module):
    def __init__(self, image_size, num_cnn, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = image_size[0]
        self.num_cnn = num_cnn

        # Build CNN block
        hidden_dim = []
        dim = 16
        for i in range(num_cnn):
            hidden_dim.append(dim)
            dim *= 2

        module = []
        input_dim = self.input_dim
        for hidden in hidden_dim:
            module.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, hidden, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden),
                    nn.LeakyReLU()
                )
            )
            input_dim = hidden

        self.input_linear = hidden_dim[-1]*image_size[1]*image_size[2]
        self.cnn_layer = nn.Sequential(*module)

        # Build linear block
        self.flatten = nn.Flatten(start_dim=1)
        self.mu = nn.Sequential(
            nn.Linear(self.input_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

        self.log_var = nn.Sequential(
            nn.Linear(self.input_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim)
        )

    def forward(self, x):
        output = self.cnn_layer(x)
        output = self.flatten(output)
        mu = self.mu(output)
        log_var = self.log_var(output)
        return (mu, log_var)

class Decoder(nn.Module):
    def __init__(self, image_size, num_cnn, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = image_size[0]
        self.output_size = tuple([-1] + list(image_size))
        
        self.num_cnn = num_cnn

        hidden_dim = []
        dim = 16
        for i in range(num_cnn):
            hidden_dim.append(dim)
            dim *= 2

        hidden_dim.reverse()
        hidden_dim.append(image_size[0])
        self.output_linear = hidden_dim[0]*image_size[1]*image_size[2]

        self.output_size = tuple([-1,hidden_dim[0]] + list(image_size)[1:])

        #build CNN block
        module = []
        output_dim = hidden_dim[1]
        for i in range(len(hidden_dim)-1):
            module.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dim[i], hidden_dim[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim[i+1]),
                nn.LeakyReLU()
            ))

        self.cnn_layer = nn.Sequential(*module)

        #Build linear layer
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_linear)
        )

    def forward(self, x):
        output = self.decoder(x)
        output = output.view(self.output_size)
        output = self.cnn_layer(output)
        return output