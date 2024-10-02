import torch
import torch.nn as nn
import torch.nn.functional as F
from hypersphere.distributions import VonMisesFisher
from hypersphere.distributions import HypersphericalUniform

class dVAE(torch.nn.Module):
    
    def __init__(self, z_dim_id=5, z_dim_emo=3, size=(64, 40), distribution_id='normal', 
        distribution_emo='vmf', kappa=1, radius=1, device="cuda"):
        """
        ModelVAE initializer
        dVAE means the double decoder VAE
        :param z_dim: dimension of the latent representation
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(dVAE, self).__init__()
        
        self.z_dim_id, self.z_dim_emo, self.size, self.device = z_dim_id, z_dim_emo, size, device
        self.distribution_id, self.distribution_emo = distribution_id, distribution_emo
        self.kappa, self.radius = kappa, radius
        
        # id encoder layers: 3 conv + 2 fc
        self.cov_en0 = nn.Conv2d(
        	in_channels=1, 
        	out_channels=16, 
        	kernel_size=(3, 3), 
        	stride=2, 
        	padding=1)
        self.pool_0 = nn.MaxPool2d(kernel_size=2)
        self.cov_en1 = nn.Conv2d(
        	in_channels=16, 
        	out_channels=64, 
        	kernel_size=(5, 5), 
        	padding="same")
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        # self.cov_en2 = nn.Conv2d(
        #     in_channels=32, 
        #     out_channels=32, 
        #     kernel_size=(5, 5), 
        #     padding="same")

        self.fc_en3 = nn.Linear(64*int(size[0]/8)*int(size[1]/8), 128)

        if self.distribution_id == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(128, z_dim_id)
            self.fc_var =  nn.Linear(128, z_dim_id)
        elif self.distribution_id == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(128, z_dim_id)
            self.fc_var = nn.Linear(128, 1)
        else:
            raise NotImplemented

        # emo encoder layers: 3 conv + 2 fc
        self.cov_en0_emo = nn.Conv2d(
            in_channels=1, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1)
        self.pool_0_emo = nn.MaxPool2d(kernel_size=2)
        self.cov_en1_emo = nn.Conv2d(
            in_channels=16, 
            out_channels=64, 
            kernel_size=(5, 5), 
            padding="same")
        self.pool_1_emo = nn.MaxPool2d(kernel_size=2)
        # self.cov_en2 = nn.Conv2d(
        #     in_channels=32, 
        #     out_channels=32, 
        #     kernel_size=(5, 5), 
        #     padding="same")

        self.fc_en3_emo = nn.Linear(64*int(size[0]/8)*int(size[1]/8), 128)

        if self.distribution_emo == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean_emo = nn.Linear(128, z_dim_emo)
            self.fc_var_emo =  nn.Linear(128, z_dim_emo)
        elif self.distribution_emo == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean_emo = nn.Linear(128, z_dim_emo)
            self.fc_var_emo = nn.Linear(128, 1)
        else:
            raise NotImplemented
            
        # decoder layers
        self.fc_de0 = nn.Linear(z_dim_id + z_dim_emo, 128)

        self.fc_de1 = nn.Linear(128, 64*int(size[0]/8)*int(size[1]/8))

        self.Tcov_de2 = nn.ConvTranspose2d(
        	in_channels=64, 
        	out_channels=32, 
        	kernel_size=(5, 5), 
        	stride=2, 
        	padding=2, 
            output_padding=1
        	)
        self.Tcov_de3 = nn.ConvTranspose2d(
            in_channels=32, 
            out_channels=16, 
            kernel_size=(3, 3), 
            stride=2, 
            padding=1, 
            output_padding=1
            )
        self.Tcov_de4 = nn.ConvTranspose2d(
        	in_channels=16, 
        	out_channels=1, 
        	kernel_size=(3, 3), 
        	stride=2, 
        	padding=1, 
            output_padding=1
        	)

        # classifier_emo
        self.linear_emo = nn.Linear(z_dim_emo, 7)
        self.softmax_class_emo = nn.Softmax(dim=1)

        # dropout
        # self.dropout = nn.Dropout(p=0.3)  # can be muted by model.eval()

    def encoder_id(self, x):
        # 2 hidden layers encoder
        x = self.pool_0(F.relu(self.cov_en0(x)))
        x = self.pool_1(F.relu(self.cov_en1(x)))
        # x = F.relu(self.cov_en2(x))
        x = x.view(-1, 64*int(self.size[0]/8)*int(self.size[1]/8))
        # x = self.dropout(x)
        x = self.fc_en3(x)  # dropout should be placed after activation
        
        if self.distribution_id == 'normal':
            # compute mean and std of the normal distribution
            z_mean_id = self.fc_mean(x)
            z_var_id = F.softplus(self.fc_var(x))
        elif self.distribution_id == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean_id = self.fc_mean(x)
            z_mean_id = self.radius * z_mean_id / z_mean_id.norm(dim=-1, keepdim=True)  # keep the points on sphere
            # the `+ 1` prevent collapsing behaviors
            z_var_id = F.softplus(self.fc_var(x))  # this is just the size of kappa
        else:
            raise NotImplemented
        
        return z_mean_id, z_var_id

    def encoder_emo(self, x):
        # 2 hidden layers encoder
        x = self.pool_0_emo(F.relu(self.cov_en0_emo(x)))
        x = self.pool_1_emo(F.relu(self.cov_en1_emo(x)))
        # x = F.relu(self.cov_en2(x))
        x = x.view(-1, 64*int(self.size[0]/8)*int(self.size[1]/8))
        # x = self.dropout(x)
        x = self.fc_en3_emo(x)  # dropout should be placed after activation
        
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
        
    def decoder(self, z):
        
        x = F.relu(self.fc_de0(z))
        x = F.relu(self.fc_de1(x))
        x = x.view(-1, 64, int(self.size[0]/8), int(self.size[1]/8))  
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
        
    def forward(self, x): 
        z_mean_id, z_var_id = self.encoder_id(x)
        z_mean_emo, z_var_emo = self.encoder_emo(x)
        # q_z_id, p_z_id = self.reparameterize(self.distribution_id, z_mean_id, z_var_id, self.z_dim_id, self.radius)
        q_z_emo, p_z_emo = self.reparameterize(self.distribution_emo, z_mean_emo, z_var_emo, self.z_dim_emo, self.radius)
        # z_id = q_z_id.rsample()  # sampling using reparameterization trick
        z_id = z_mean_id
        z_emo = q_z_emo.rsample()

        x_rec = self.decoder(torch.cat((z_id, z_emo), 1))
        x_cla_emo = self.classifier_emo(z_emo)
        
        return (z_mean_id, z_mean_emo), (q_z_emo, p_z_emo), (z_id, z_emo), (x_rec, x_cla_emo)

    
