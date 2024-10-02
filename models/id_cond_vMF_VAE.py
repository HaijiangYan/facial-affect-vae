import torch
import torch.nn as nn
import torch.nn.functional as F
from hypersphere.distributions import VonMisesFisher
from hypersphere.distributions import HypersphericalUniform

class conditionalVAE(torch.nn.Module):
    
    def __init__(self, z_dim_emo=3, size=(64, 40), distribution_emo='vmf', kappa=1, radius=1, device="cuda"):
        """
        ModelVAE initializer
        dVAE means the double decoder VAE
        :param z_dim: dimension of the latent representation
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(conditionalVAE, self).__init__()
        
        self.z_dim_emo, self.size, self.device = z_dim_emo, size, device
        self.distribution_emo = distribution_emo
        self.kappa, self.radius = kappa, radius

        # emo encoder layers: 3 conv + 2 fc
        self.cov_en0_emo = nn.Conv2d(
            in_channels=1, 
            out_channels=8, 
            kernel_size=(3, 3), 
            padding="same")
        self.pool_0_emo = nn.MaxPool2d(kernel_size=2)
        self.cov_en1_emo = nn.Conv2d(
            in_channels=8, 
            out_channels=16, 
            kernel_size=(3, 3), 
            padding="same")
        self.pool_1_emo = nn.MaxPool2d(kernel_size=2)
        self.cov_en2_emo = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(5, 5), 
            padding="same")
        self.pool_2_emo = nn.MaxPool2d(kernel_size=2)

        self.fc_en3_emo = nn.Linear(32*int(size[0]/8)*int(size[1]/8), 16)

        if self.distribution_emo == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean_emo = nn.Linear(16, z_dim_emo)
            self.fc_var_emo =  nn.Linear(16, z_dim_emo)
        elif self.distribution_emo == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean_emo = nn.Linear(16, z_dim_emo)
            self.fc_var_emo = nn.Linear(16, 1)
        else:
            raise NotImplemented
            
        # decoder layers
        self.fc_de0 = nn.Linear(1 + z_dim_emo, 16)

        self.fc_de1 = nn.Linear(16, 32*int(size[0]/8)*int(size[1]/8))

        self.Tcov_de2 = nn.ConvTranspose2d(
        	in_channels=32, 
        	out_channels=16, 
        	kernel_size=(5, 5), 
        	stride=2, 
        	padding=2, 
            output_padding=1
        	)
        self.Tcov_de3 = nn.ConvTranspose2d(
            in_channels=16, 
            out_channels=8, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1, 
            output_padding=1
            )
        self.Tcov_de4 = nn.ConvTranspose2d(
        	in_channels=8, 
        	out_channels=1, 
        	kernel_size=(3, 3), 
        	stride=2, 
        	padding=1, 
            output_padding=1
        	)

        # classifier_emo
        self.dropout = nn.Dropout(0.7)
        self.linear_emo = nn.Linear(z_dim_emo, 7)
        self.softmax_class_emo = nn.Softmax(dim=1)

    def encoder_emo(self, x):
        # 2 hidden layers encoder
        x = self.pool_0_emo(F.relu(self.cov_en0_emo(x)))
        x = self.pool_1_emo(F.relu(self.cov_en1_emo(x)))
        x = self.pool_2_emo(F.relu(self.cov_en2_emo(x)))
        x = x.view(-1, 32*int(self.size[0]/8)*int(self.size[1]/8))
        x = self.dropout(x)
        x = self.fc_en3_emo(x)  # dropout should be placed after activation
        x = self.dropout(x)
        
        if self.distribution_emo == 'normal':
            # compute mean and std of the normal distribution
            z_mean_emo = self.fc_mean_emo(x)
            z_var_emo = F.softplus(self.fc_var_emo(x))
        elif self.distribution_emo == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean_emo = self.fc_mean_emo(x)
            z_mean_emo = self.radius * z_mean_emo / z_mean_emo.norm(dim=-1, keepdim=True)  # keep the points on sphere
            # the `+ 1` prevent collapsing behaviors
            z_var_emo = F.softplus(self.fc_var_emo(x))  # this is just the size of kappa
        else:
            raise NotImplemented
        
        return z_mean_emo, z_var_emo

    def classifier_emo(self, z):

        x = self.linear_emo(z)
        x = self.softmax_class_emo(x)

        return x
        
    def decoder(self, x_id, z):
        
        x = F.relu(self.fc_de0(torch.cat((x_id[:, None], z), 1)))
        x = F.relu(self.fc_de1(x))
        x = x.view(-1, 32, int(self.size[0]/8), int(self.size[1]/8))  
        # dimensionality transform: batch, channel, size...
        x = F.relu(self.Tcov_de2(x))
        x = F.relu(self.Tcov_de3(x))
        x = F.sigmoid(self.Tcov_de4(x))
        # x = self.Tcov_de4(x)
        
        return x
        
    def reparameterize(self, distribution, z_mean, z_var, z_dim, radius):
        if distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var, k=self.kappa, radius=radius)  # new real kappa is here
            p_z = HypersphericalUniform(z_dim - 1, radius, device=self.device)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x_id, x): 
        z_mean_emo, z_var_emo = self.encoder_emo(x)
        q_z_emo, p_z_emo = self.reparameterize(self.distribution_emo, z_mean_emo, z_var_emo, self.z_dim_emo, self.radius)
        # z_id = q_z_id.rsample()  # sampling using reparameterization trick
        z_emo = q_z_emo.rsample()

        x_rec = self.decoder(x_id, z_emo)
        x_cla_emo = self.classifier_emo(z_mean_emo)
        
        return (z_mean_emo, z_var_emo), (q_z_emo, p_z_emo), z_emo, (x_rec, x_cla_emo)

    

    
