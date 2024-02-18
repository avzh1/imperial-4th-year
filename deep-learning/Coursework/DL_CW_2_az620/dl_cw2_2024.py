#!/usr/bin/env python
# coding: utf-8

# # Coursework 2: Generative Models
# 
# ## Instructions
# 
# ### Submission 
# Please submit one zip file on cate - *CW2.zip* containing the following:
# 1. A version of this notebook containing your answers. Write your answers in the cells below each question. **Please deliver the notebook including the outputs of the cells**
# 2. Your trained VAE model as *VAE_model.pth*
# 3. Your trained Generator and Discriminator: *DCGAN_model_D.pth and DCGAN_model_G.pth*
# 
# 
# ### Training
# Training the GAN will take quite a long time (multiple hours), please refer to the 4 GPU options detailed in the logistics lecture. Some additional useful pointers:
# * PaperSpace [guide if you need more compute](https://hackmd.io/@afspies/S1stL8Qnt)
# * Lab GPUs via SSH.  The VSCode Remote Develop extension is recommended for this. For general Imperial remote working instructions see [this post](https://www.doc.ic.ac.uk/~nuric/teaching/remote-working-for-imperial-computing-students.html). You'll also want to [setup your environment as outlined here](https://hackmd.io/@afspies/Bkd7Zq60K).
# * Use Colab and add checkpointing to the model training code; this is to handle the case where colab stops a free-GPU kernel after a certain number of hours (~4).
# * Use Colab Pro - If you do not wish to use PaperSpace then you can pay for Colab Pro. We cannot pay for this on your behalf (this is Google's fault).
# 
# 
# ### Testing
# TAs will run a testing cell (at the end of this notebook), so you are required to copy your data ```transform``` and ```denorm``` functions to a cell near the bottom of the document (it is demarkated). You are advised to check that your implementations pass these tests (in particular, the jit saving and loading may not work for certain niche functions)
# 
# ### General
# You can feel free to add architectural alterations / custom functions outside of pre-defined code blocks, but if you manipulate the model's inputs in some way, please include the same code in the TA test cell, so our tests will run easily.
# 
# <font color="orange">**The deadline for submission is Monday, 26 Feb by 6 pm** </font>

# ## Setting up working environment
# You will need to install pytorch and import some utilities by running the following cell:

# In[15]:


# !pip install -q torch torchvision altair seaborn tqdm numpy matplotlib torchsummary
# !git clone -q https://github.com/afspies/icl_dl_cw2_utils
from icl_dl_cw2_utils.utils.plotting import plot_tsne
from pathlib import Path

from mpl_toolkits.axes_grid1 import ImageGrid

import os
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchsummary import summary


# In[16]:


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Here we have some default pathing options which vary depending on the environment you are using. You can of course change these as you please.

# In[17]:


# Initialization Cell
WORKING_ENV = 'HOME'
USERNAME = 'az620' # If working on Lab Machines - Your college username
assert WORKING_ENV in ['LABS', 'COLAB', 'PAPERSPACE', 'SAGEMAKER', 'HOME']

if WORKING_ENV == 'COLAB':
    from google.colab import drive
    get_ipython().run_line_magic('load_ext', 'google.colab.data_table')
    dl_cw2_repo_path = 'dl_cw2/' # path in your gdrive to the repo
    content_path = f'/content/drive/MyDrive/{dl_cw2_repo_path}' # path to gitrepo in gdrive after mounting
    data_path = './data/' # save the data locally
    drive.mount('/content/drive/') # Outputs will be saved in your google drive

elif WORKING_ENV == 'LABS':
    content_path = f'/vol/bitbucket/{USERNAME}/dl/dl_cw2/' # You may want to change this
    data_path = f'/vol/bitbucket/{USERNAME}/dl/'
    # Your python env and training data should be on bitbucket
    if 'vol/bitbucket' not in content_path or 'vol/bitbucket' not in data_path:
        import warnings
        warnings.warn(
           'It is best to create a dir in /vol/bitbucket/ otherwise you will quickly run into memory issues'
           )
elif WORKING_ENV == 'PAPERSPACE': # Using Paperspace
    # Paperspace does not properly render animated progress bars
    # Strongly recommend using the JupyterLab UI instead of theirs
    get_ipython().system('pip install ipywidgets')
    content_path = '/notebooks/'
    data_path = './data/'
    
elif WORKING_ENV == 'SAGEMAKER':
    content_path = '/home/studio-lab-user/sagemaker-studiolab-notebooks/dl/'
    data_path = f'{content_path}data/'

elif WORKING_ENV == 'HOME':
    content_path = '/home/avzh1/Documents/imperial/year4/lectures/deep-learning/Coursework/DL_CW_2_az620/content/'
    data_path = f'{content_path}data/'
else:
  raise NotImplementedError()

content_path = Path(content_path)


# In[18]:


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

if not os.path.exists(content_path/'CW_VAE/'):
    os.makedirs(content_path/'CW_VAE/')

if not os.path.exists(data_path):
    os.makedirs(data_path)

# We set a random seed to ensure that your results are reproducible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')


# ## Introduction
# 
# For this coursework, you are asked to implement two commonly used generative models:
# 1. A **Variational Autoencoder (VAE)**
# 2. A **Deep Convolutional Generative Adversarial Network (DCGAN)**
# 
# For the first part you will the MNIST dataset https://en.wikipedia.org/wiki/MNIST_database and for the second the CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html).
# 
# Each part is worth 50 points. 
# 
# The emphasis of both parts lies in understanding how the models behave and learn, however, some points will be available for getting good results with your GAN (though you should not spend too long on this).

# # Part 1 - Variational Autoencoder

# ## Part 1.1 (25 points)
# **Your Task:**
# 
# a. Implement the VAE architecture with accompanying hyperparameters. More marks are awarded for using a Convolutional Encoder and Decoder.
# 
# b. Design an appropriate loss function and train the model.

# ---
# ### Part 1.1a: Implement VAE (25 Points)

# #### Hyper-parameter selection

# In[19]:


# Necessary Hyperparameters 
num_epochs = 32
learning_rate = 0.001
batch_size = 128
latent_dim = 2 # Choose a value for the size of the latent space

# Additional Hyperparameters 
beta = 2
hidden_dimensions = 16


# #### Data loading
# 

# In[20]:


# Transformed data is first padded and randomly cropped to encourage the encoder to memorize patterns, not just
# centered location.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(32,32))
    # transforms.RandomResizedCrop(size=(32,32)),
    # transforms.Normalize(mean=0, std=1)
])

# (Optionally) Modify the network's output for visualizing your images
def denorm(x):
    return x


# In[21]:


train_dat = datasets.MNIST(data_path, train=True, download=True, transform=transform)
test_dat = datasets.MNIST(data_path, train=False, transform=transform)

loader_train = DataLoader(train_dat, batch_size, shuffle=True)
loader_test = DataLoader(test_dat, batch_size, shuffle=False)

# Don't change 
sample_inputs, _ = next(iter(loader_test))
fixed_input = sample_inputs[:32, :, :, :]
save_image(fixed_input, content_path/'CW_VAE/image_original.png')


# In[22]:


# Get a batch of data
sample_inputs, _ = next(iter(loader_train))
fixed_input = sample_inputs[:64, :, :, :]

# Create a grid of images
grid_img = make_grid(fixed_input, nrow=8, padding=2, normalize=True)

# Display the grid using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.show()


# #### Model Definition

# 
# <figure>
#   <img src="https://blog.bayeslabs.co/assets/img/vae-gaussian.png" style="width:60%">
#   <figcaption>
#     Fig.1 - VAE Diagram (with a Guassian prior), taken from <a href="https://blog.bayeslabs.co/2019/06/04/All-you-need-to-know-about-Vae.html">1</a>.
#   </figcaption>
# </figure>
# 
# Hints:
# - It is common practice to encode the log of the variance, rather than the variance
# - You might try using BatchNorm

# For the encoder, I chose to implement the ResNet architecture due to its repeated success in classification tasks. My hope, is that for similar reasons, this architecture will prove to be most effective in an encoder context.

# In[23]:


class EncoderBlock_DEPRECATED(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size=3,
                       stride=1): 
        """Generate an encoder block for the VAE in a ResNet style. 

        init > Conv2d > BatchNorm > ReLU > Conv2d > BatchNorm := right
        init > 1x1Conv > BatchNorm := left
        
        right + left > ReLU

        Args:
            in_channels (int): number of input dimensions (_, in_channels, _, _) 
            out_channels (int): number of output dimensions to output
            kernel_size (int, optional): Defaults to 3.
            stride (int, optional): Defaults to 1.
        """
        
        
        super(EncoderBlock_DEPRECATED, self).__init__()
        
        # input is (_, in, height, width)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=1
        )

        # input is (_, out, height', width')
        # height' = (height-kernel+2) / stride + 1
        # width' similarly

        self.norm1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # This convolutional layer cannot change the dimension of the output image; hard code parameters here.

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1
        )

        # input is (_, out, height', width')

        self.norm2 = nn.BatchNorm2d(num_features=out_channels)

        # input is (_, in, height, width)

        self.shortcut = nn.Conv2d(in_channels=in_channels, 
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0
        )

        # input is (_, out, height, width)

        self.norm3 = nn.BatchNorm2d(num_features=out_channels)

        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass on the ResNet block

        Args:
            x (torch.tensor): of dimension (batch_size, in_channels, height, width)

        Returns:
            torch.tensor: (batch_size, out_channels, height', width')
        """
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.relu1(c)
        c = self.conv2(c)
        c = self.norm2(c)

        i = self.shortcut(x)   
        i = self.norm3(i)

        out = c + i

        return self.relu2(out)


# In[24]:


class DecoderBlock_DEPRECATED(nn.Module):
    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size=3,
                       stride=2,
                       padding=1): 
        """Generate an decoder block for the VAE. 

        Args:
            in_channels (int): number of input dimensions (_, in_channels, _, _) 
            out_channels (int): number of output dimensions to output
            kernel_size (int, optional): Defaults to 3.
            stride (int, optional): Defaults to 1.
        """
        
        
        super(DecoderBlock_DEPRECATED, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels, 
                                          kernel_size=3,
                                          stride=2, 
                                          padding=1,
                                          output_padding=1)
        
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(in_channels=out_channels,
                                          out_channels=out_channels, 
                                          kernel_size=3,
                                          stride=1, 
                                          padding=1,
                                          output_padding=0)

        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """Forward pass on the decoder network block

        Args:
            x (torch.tensor): of dimension (batch_size, in_channels, height, width)

        Returns:
            torch.tensor: (batch_size, out_channels, height', width')
        """
        
        x = self.deconv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x


# In[25]:


class VAE_DEPRECATED(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_DEPRECATED, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Encoder Modules

        # (_, 1, 28, 28) -> (_, 16, 28, 28)
        self.encode1 = EncoderBlock_DEPRECATED(in_channels=1,
                                    out_channels=(hidden_dimensions//8),
                                    kernel_size=3,
                                    stride=1
        )

        # (_, 16, 28, 28) -> (_, 32, 14, 14)
        self.encode2 = EncoderBlock_DEPRECATED(in_channels=(hidden_dimensions//8),
                                    out_channels=(hidden_dimensions//8)*2,
                                    kernel_size=3,
                                    stride=2)

        # (_, 32, 14, 14) -> (_, 64, 7, 7)
        self.encode3 = EncoderBlock_DEPRECATED(in_channels=(hidden_dimensions//8)*2,
                                    out_channels=(hidden_dimensions//8)*5,
                                    kernel_size=3,
                                    stride=2)
        
        # (_, 64, 7, 7) -> (_, 128, 4, 4)
        self.encode4 = EncoderBlock_DEPRECATED(in_channels=(hidden_dimensions//8)*5,
                                    out_channels=hidden_dimensions,
                                    kernel_size=3,
                                    stride=2)

        # # Image grid to single flat vector
        self.latent  = nn.Flatten()

        # Latent Mean and Variance 
        self.mean_layer   = nn.Linear(hidden_dimensions * 4 * 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dimensions * 4 * 4, latent_dim)

        # Decoder Modules

        # From linear latent space back into image form
        self.up_sample1 = nn.Linear(latent_dim, hidden_dimensions * 4 * 4) # don't forget to input.reshape(input.shape[0], -1, 4, 4)
       
        # (_, 1024, 4, 4) -> (_, 768, 7, 7)
        self.decode1 = DecoderBlock_DEPRECATED(in_channels=hidden_dimensions,
                                    out_channels=(hidden_dimensions//8)*5,
                                    kernel_size=2,
                                    stride=2)
        
        # (_, 768, 7, 7) -> (_, 512, 13, 13)
        self.decode2 = DecoderBlock_DEPRECATED(in_channels=(hidden_dimensions//8)*5,
                                    out_channels=(hidden_dimensions//8)*2, 
                                    kernel_size=2,
                                    stride=2)

        # (_, 512, 13, 13) -> (_, 256, 25, 25)
        self.decode3 = DecoderBlock_DEPRECATED(in_channels=(hidden_dimensions//8)*2, 
                                    out_channels=1, 
                                    kernel_size=2,
                                    stride=2)

        self.sigmoid = nn.Sigmoid()
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
    def encode(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)

        x = self.latent(x)

        return self.mean_layer(x), self.logvar_layer(x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
    
    def reparametrize(self, mu, logvar):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        epsilon = torch.randn_like(logvar)#.to(device)
        sampled_latent = mu + logvar*epsilon
        return sampled_latent

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 

    def decode(self, z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        z = self.up_sample1(z)

        z = z.reshape(z.shape[0], -1, 4, 4)

        z = self.decode1(z)
        z = self.decode2(z)
        z = self.decode3(z)

        return self.sigmoid(z)

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
    
    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)

        return decoded, mu, logvar

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 

model = VAE_DEPRECATED(latent_dim).to(device)

summary(model, (1, 32, 32))

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))


# In[26]:


# *CODE FOR PART 1.1a IN THIS CELL*

class VAE(nn.Module):
    """VAE module for MNIST dataset"""
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # Encoder Modules
        self.encoder = nn.Sequential(
            # (_, 1, 32, 32) -> (_, 16, 32, 32)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # (_, 16, 32, 32) -> (_, 16, 16, 16)
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            # (_, 16, 16, 16) -> (_, 64, 16, 16)
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (_, 64, 16, 16) -> (_, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # (_, 64, 8, 8,) -> (_, 4096)
            nn.Flatten(),

            # (_, 4096) -> (_, 1024)
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            
            # (_, 1024) -> (_, 1024)
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        )

        # Latent space mean and variance layers
        self.mean_layer   = nn.Linear(1024, latent_dim)
        self.logvar_layer = nn.Linear(1024, latent_dim)

        self.decoder = nn.Sequential(
            # (_, latent_dim) -> (_, 1024)
            nn.Linear(in_features=latent_dim, out_features=1024),
            nn.ReLU(),


            # (_, 1024) -> (_, 1024)
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),

            # (_, 1024) -> (_, 4096)
            nn.Linear(in_features=1024, out_features=4096),
            nn.ReLU(),

            # (_, 4096) -> (_, 64, 8, 8)
            nn.Unflatten(dim=1, unflattened_size=(64, 8, 8)),

            # (_, 64, 8, 8) -> (_, 16, 16, 16)
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # (_, 16, 16, 16) -> (_, 1, 32, 32)
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.Sigmoid()
        )
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
        
    def encode(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        x = self.encoder(x)

        return self.mean_layer(x), self.logvar_layer(x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
    
    def reparametrize(self, mu, logvar):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        epsilon = torch.randn_like(logvar)#.to(device)
        sampled_latent = mu + logvar*epsilon
        return sampled_latent

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 

    def decode(self, z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        return self.decoder(z)

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 
    
    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        decoded = self.decode(z)

        return decoded, mu, logvar

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 

model = VAE(latent_dim).to(device)

summary(model, (1, 32, 32))

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))


# --- 
# 
# ### Part 1.1b: Training the Model (5 Points)

# #### Defining a Loss
# Recall the Beta VAE loss, with an encoder $q$ and decoder $p$:
# $$ \mathcal{L}=\mathbb{E}_{q_\phi(z \mid X)}[\log p_\theta(X \mid z)]-\beta D_{K L}[q_\phi(z \mid X) \| p_\theta(z)]$$
# 
# $$ D_{KL}[p||q]=\mathbb E_{p(\vec x)} \big[ \log \frac{p(\vec x)}{q (\vec x)}\big] = \int p(\vec x) \log \frac{p(\vec x)}{q(\vec x)}d \vec x $$
# 
# In order to implement this loss you will need to think carefully about your model's outputs and the choice of prior.
# 
# There are multiple accepted solutions. Explain your design choices based on the assumptions you make regarding the distribution of your data.
# 
# * Hint: this refers to the log likelihood as mentioned in the tutorial. Make sure these assumptions reflect on the values of your input data, i.e. depending on your choice you might need to do a simple preprocessing step.
# 
# * You are encouraged to experiment with the weighting coefficient $\beta$ and observe how it affects your training

# In[27]:


def loss_function_VAE(recon_x, x, mu, logvar, beta):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kl_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = reconstruction_loss + beta * kl_loss

        return loss, reconstruction_loss, kl_loss

        #######################################################################
        #                       ** END OF YOUR CODE **
        ####################################################################### 


# In[28]:


import logging

def train_model(b = 0, ld=2, lr=0.001):
    
    def train_model():
        """Main training function"""
        # Setup model
        # my_logger.info('======================================================')
        # my_logger.info('Beginning training for model')
        # my_logger.info('======================================================')
        
        global beta
        beta = b
        global latent_dim
        latent_dim = ld

        model = VAE(latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training mode
        model.train()

        per_epoch_loss = []
        per_epoch_reconstruction_loss = []
        per_epoch_kl_loss = []

        per_epoch_loss_test = []
        per_epoch_reconstruction_loss_test = []
        per_epoch_kl_loss_test = []

        for epoch in range(num_epochs):

            # my_logger.info(f"Training - Epoch {epoch + 1}/{num_epochs}")
            
            loss, reconstruction_loss, kl_loss = 0.0, 0.0, 0.0

            # Training
            with tqdm.tqdm(loader_train, unit="batch") as tepoch:
                for batch_idx, (data, _) in enumerate(tepoch):
                    
                    data = data.to(device)
                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(data)

                    loss, reconstruction_loss, kl_loss = loss_function_VAE(recon_batch, data, mu, logvar, beta)

                    loss.backward()
                    optimizer.step()

                    tepoch.set_description(f"Epoch {epoch + 1}")
                    tepoch.set_postfix(loss=loss.item() / len(data))
                        
                    # Log the information
                    # my_logger.info(f"Epoch {epoch+1} - Iteration {batch_idx + 1}/{len(loader_train)} - Loss: {loss.item() / len(data)}, Reconstruction Loss: {reconstruction_loss.item() / len(data)}, KL Loss: {kl_loss.item() / len(data)}")

                    loss = loss.item() / len(data)
                    reconstruction_loss = reconstruction_loss.item() / len(data)
                    kl_loss = kl_loss.item() / len(data)
                
            per_epoch_loss.append(loss)
            per_epoch_reconstruction_loss.append(reconstruction_loss)
            per_epoch_kl_loss.append(kl_loss)

            # Testing
            model.eval()  # Switch to evaluation mode
            test_loss = 0.0
            test_reconstruction_loss = 0.0
            test_kl_loss = 0.0
            with torch.no_grad():
                for data_test, _ in loader_test:
                    data_test = data_test.to(device)
                    recon_batch_test, mu_test, logvar_test = model(data_test)

                    loss_test, reconstruction_loss_test, kl_loss_test = loss_function_VAE(recon_batch_test, data_test, mu_test, logvar_test, beta)
                    
                    test_loss += loss_test.item()
                    test_reconstruction_loss += reconstruction_loss_test.item()
                    test_kl_loss += kl_loss_test.item()

            test_loss /= len(loader_test.dataset)
            test_reconstruction_loss /= len(loader_test.dataset)
            test_kl_loss /= len(loader_test.dataset)
            
            # Log the information
            # my_logger.info(f"Epoch {epoch+1} - Test Loss: {test_loss}, Test Reconstruction Loss: {test_reconstruction_loss}, Test KL Loss: {test_kl_loss}")

            per_epoch_loss_test.append(test_loss)
            per_epoch_reconstruction_loss_test.append(test_reconstruction_loss)
            per_epoch_kl_loss_test.append(test_kl_loss)
            
            model.train()  # Switch back to training mode

            # # Save the model at the end of each epoch
            # if epoch == num_epochs - 1:
            #     with torch.no_grad():
            #         torch.jit.save(torch.jit.trace(model, (data), check_trace=False), f'{logger_path}/VAE_model.pth')
    train_model()  

    # Clear logger
    # my_logger.removeHandler(my_log_handler)


# In[30]:


betas = [2] # [0, 0.1, 1, 2, 3]
lds = [5] # [2, 5, 10]
lrs = [0.01] # [0.1, 0.01, 0.001]

for bs_ in betas:
    for lds_ in lds:
        for lrs_ in lrs:
            train_model(b=bs_, ld=lds_, lr=lrs_)
