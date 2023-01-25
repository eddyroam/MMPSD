import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """
    Conditional VAE is trained on the entire dataset, with the class label embedded 
    such that synthetic data of specific class can be generated
    """
    def __init__(self):
        super(ConditionalVAE, self).__init__()
        self.features = 50 #Latent vector size
        self.labels = 10 #Label encoding size
 
        # encoder
        self.enc1 = nn.Linear(in_features=4000, out_features=500)
        self.enc2 = nn.Linear(in_features=500, out_features=500)
        self.enc3 = nn.Linear(in_features=500, out_features=self.features*2)
        self.enc4 = nn.Linear(in_features=(self.features*2 + self.labels), out_features=self.features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=(self.features + self.labels), out_features=self.features)
        self.dec2 = nn.Linear(in_features=self.features, out_features=500)
        self.dec3 = nn.Linear(in_features=500, out_features=500)
        self.dec4 = nn.Linear(in_features=500, out_features=4000)
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    def encoder(self, x, y):
        """
        :x: input data of size 4000
        :y: label encoding of size 10
        """
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = torch.cat([x, y], dim=1)
        x = self.enc4(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
       
        return mu, log_var
    
    def decoder(self, z, y):
        """
        :y: label encoding of size 10
        :z: Reparametized variable
        """
        z = torch.cat([z, y], dim=1)
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction
    
    def forward(self, x, y):
        # encoding
        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
 
        # decoding
        reconstruction = self.decoder(z, y)
        return reconstruction, mu, log_var
 

class IndividualVAE(nn.Module):
    """
    Individual VAE is trained only on dataset for one class, each VAE being used to
    generate synthetic data of a specific class
    """
    def __init__(self):
        super(IndividualVAE, self).__init__()
        self.features = 50 #Latent vector size
 
        # encoder
        self.enc1 = nn.Linear(in_features=4000, out_features=500)
        self.enc2 = nn.Linear(in_features=500, out_features=500)
        self.enc3 = nn.Linear(in_features=500, out_features=self.features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=self.features, out_features=500)
        self.dec2 = nn.Linear(in_features=500, out_features=500)
        self.dec3 = nn.Linear(in_features=500, out_features=4000)
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
    
    def encoder(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x).view(-1, 2, self.features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        return mu, log_var
    
    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        reconstruction = torch.sigmoid(self.dec3(x))
        return reconstruction
 
    def forward(self, x):
        # encoding
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
 
        # decoding
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var