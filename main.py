#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# 

# In[2]:


# DOWNLOAD THE VGG16 MODEL
import torchvision.models as models
vgg16=models.vgg16(pretrained=True)


# In[3]:


dataroot = "./train/"
# DATASET OF 20K IMAGES FOR TRAINING


# In[4]:


#PARAMETERS FOR LOADING DATASET

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this.
image_size = 64
image_size_224 = 224

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 699

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# In[5]:


# Learning rate for optimizers
lr = 0.001

# Beta hyperparam for Adam optimizers
beta1 = 0.9
beta2 = 0.999


# In[6]:


# Create the dataset, dataset stores 64*64*3 images and 224*224*3 images
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_224= dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size_224),
                               transforms.CenterCrop(image_size_224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader for both datasets
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
dataloader_224 = torch.utils.data.DataLoader(dataset_224, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# real_batch = next(iter(dataloader_224))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:224], padding=2, normalize=True).cpu(),(1,2,0)))


# In[7]:


print("Running on: ",device)


# In[8]:


#first 14 layers of vgg16
first_14_fun_cuda=lambda x:vgg16.classifier.train(mode=False).to(device)[:2](vgg16.avgpool(vgg16.features.train(mode=False).to(device)(x)).to(device).detach().reshape(-1,25088))
def phi(image):
  ftr_img=first_14_fun_cuda(image.to(device)).cpu().detach()
  return ftr_img


# In[9]:


#TEST CELL
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
fixed_noise = torch.randn(2, 3, 224, 224, device=device)
test=phi(fixed_noise).reshape(-1,4096)
assert test.shape==(2,4096)


# In[10]:


#Eps returns the relu3_3 output of vgg16 model
E=lambda x:vgg16.features.train(mode=False).cuda()[:16](x)
def Eps(image):
  ftr_img=E(image.cuda()).cpu()
  return ftr_img
phi_Eps_16=lambda x:vgg16.classifier.train(mode=False).to(device)[:2](vgg16.avgpool(vgg16.features[16:].train(mode=False).to(device)(x)).to(device).detach().reshape(-1,25088))
#phi_Eps applies on the Eps output of an image to generate the phi output
def phi_Eps(image):
  ftr_img=phi_Eps_16(image.to(device)).cpu().detach()
  return ftr_img


# In[11]:


#TEST CELL
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
fixed_noise = torch.randn(2, 3, 224, 224, device=device)
test=phi_Eps(Eps(fixed_noise))
test2=phi(fixed_noise)
assert np.allclose(test.numpy(),test2.numpy())


# In[12]:


#returns a matrix which can be used to apply PCA on PyTorch Tensors
def PCA_fit(X, k=2):
 # preprocess the data
 X_mean = torch.mean(X,0)
 X = X - X_mean.expand_as(X)
 # svd
 U,S,V = torch.svd(torch.t(X))
 return U[:,:k]
# Takes as input a tensor and pca_mat to generate the transformed tensor
def PCA_transform(X,U):
  return torch.mm(X,U)


# In[13]:


### Can be used to generate the pca_mat matrix
# i=0
# first=False
# for batch in dataloader_224:
#   if first==False:
#     first=True
#     phi_x=phi(batch[0]).reshape(-1,4096)
#   else:
#     phi_x=torch.cat((phi_x,phi(batch[0]).reshape(-1,4096)))
#   if i%32==0:
#     print(i)
#   i=i+1
# PATH_PCAMAT='./CHECKPOINT/pcamat.pth'
# pca_mat=PCA_fit(phi_x,k=699)
# torch.save({'pcamat':pca_mat},PATH_PCAMAT)


# In[14]:


PATH_PCAMAT='./CHECKPOINT/pcamat.pth'
checkpoint=torch.load(PATH_PCAMAT)
pca_mat=checkpoint['pcamat']


# 

# In[15]:


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        # WE USED THIS FOR FASTER CONVERGENCE 
        # ACTUAL PAPER USES: nn.init.normal_(m.weight.data, 1.0, 0.01) 
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[16]:


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# In[17]:


# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)


# In[18]:


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# In[19]:


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
netD.apply(weights_init)


# In[20]:


# Initialize Loss functions
criterion = nn.BCELoss()
crit_mse = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 0.99 ##
fake_label = 0.01 ##

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))


# In[21]:


#INITIALISATION CODE
random.seed(1221)
netG.apply(weights_init)
netD.apply(weights_init)
# # Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
LAMBDA_DIS=100
LAMBDA_ADV=100
LAMBDA_FEA=0.01
LAMBDA_STI=2e-6
epoch_ck=0
upsampler = nn.UpsamplingBilinear2d(size=(224, 224))


# In[22]:


SV_PATH='./CHECKPOINT/savedmodel.pth'


# In[25]:


## LOAD WEIGHTS OF PREVIOUS TRAINING SESSION
LD_PATH='./CHECKPOINT/pretrained_stimuli_6L.pth'
checkpoint = torch.load(LD_PATH)
netG.load_state_dict(checkpoint['modelG_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
netD.load_state_dict(checkpoint['modelD_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
epoch_ck = checkpoint['epoch']


# In[24]:


# Training Loop
saveNumber=0
print("Starting Training Loop...")
# For each epoch
for epoch in range(epoch_ck,num_epochs):
    # For each batch in the dataloader
    for i, dataTuple in enumerate(zip(dataloader,dataloader_224), 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
       
        
        data=dataTuple[0]
        data_224=dataTuple[1]
        
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        real_cpu_224 = data_224[0].to(device)
        b_size = real_cpu_224.size(0)
        
        # Generate batch of latent vectors
        with torch.no_grad():
            eps_real_cpu_224=Eps(real_cpu_224)
            phi_real_cpu_224=phi_Eps(eps_real_cpu_224)
            z=PCA_transform(phi_real_cpu_224,pca_mat)
            z=z.reshape(z.shape[0],z.shape[1],1,1).to(device)
            noise=z
        
        netD.zero_grad() # Init gradients to 0
        ## Train with all-real batch
        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_cpu).view(-1) # Forward pass real batch through D
        errD_real = LAMBDA_DIS*criterion(output, label) # Calculate loss on all-real batch
        errD_real.backward() # Calculate gradients for D in backward pass
        D_x = output.mean().item() # For tracking progress

        ## Train with all-fake batch        
        fake = netG(noise) # Generate fake image batch with G
        label.fill_(fake_label) # Classify all fake batch with D
        output = netD(fake.detach()).view(-1) # Calculate D's loss on the all-fake batch
        errD_fake = LAMBDA_DIS*criterion(output, label) # Calulate error for this batch
        errD_fake.backward() # Calculate the gradients for this batch
        D_G_z1 = output.mean().item() # For tracking progress
        
        errD = errD_real + errD_fake # Add the gradients from the all-real and all-fake batches
        optimizerD.step() # Update D

        ############################
        # (2) Update G network: maximize log(D(G(z))) + LOSS_FEA + LOSS_STI
        ###########################
        netG.zero_grad()
        
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)# Since we just updated D, perform another forward pass of all-fake batch through D
        errG_a = LAMBDA_ADV * criterion(output, label)# Calculate G's loss based on this output
        errG_a.backward()# Calculate gradients for G
        
        
        G_z=netG(noise.detach()) # Generate an image
        G_z_224=upsampler(G_z)  # Upsample it to pass through vgg16
        eps_G_z_224=Eps(G_z_224)# Get features of Generator output 
        errG_fea= LAMBDA_FEA*crit_mse(eps_real_cpu_224,eps_G_z_224)# Get feature loss
        errG_fea.backward()# Backward pass 
        
        G_z2=netG(noise.detach()) # Generate an image
        errG_sti = LAMBDA_STI*crit_mse(real_cpu,G_z2)# Get stimuli loss
        errG_sti.backward() #Backward Pass

        errG=errG_a+errG_fea+errG_sti # Total loss
        D_G_z2 = output.mean().item() # For tracking progress
        
        optimizerG.step() # Update parameters of Generator

        # Output training stats
        if i % 100  == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f / %.4f\tLoss_G:   %.4f / %.4f \tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD_fake.item(), errD_real.item(), errG_fea.item(),errG_sti.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        # Periodically save weights for retraining
        if(iters%1000==0):
            torch.save({
            'epoch': epoch,
            'modelG_state_dict': netG.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'errG': errG,
            'modelD_state_dict': netD.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'errD': errD,
            }, SV_PATH)
            saveNumber += 1
        iters += 1


# In[26]:


# EVALUATION
# LOAD TEST DATASET
dataset = dset.ImageFolder(root='./test/',
                           transform=transforms.Compose([
                               transforms.Resize(image_size_224),
                               transforms.CenterCrop(image_size_224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)


# In[43]:


#EVALUATION METRICS
import kornia
def struct_similiarity(x,y):
    ssim=kornia.losses.SSIM(3)
    loss=torch.mean(ssim(x,y))
    return loss
def pearson_coefficient(x,y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost
featureExtractor=lambda x:vgg16.classifier[:5].train(mode=False).to(device)[:2](vgg16.avgpool(vgg16.features.train(mode=False).to(device)(x)).to(device).detach().reshape(-1,25088))
def feature_similarity(x,y):
    f1=featureExtractor(x)
    f2=featureExtractor(y)
    euclidean = nn.PairwiseDistance(p=2)
    cost = euclidean(f1,f2).mean()
    return cost


# In[55]:


structure=np.array([])
pearson=np.array([])
feature=np.array([])

for i, data in enumerate(dataloader_test, 0):
    batch=data[0].to(device)
    with torch.no_grad():
        z=PCA_transform(phi(batch),pca_mat)
        z=z.reshape(z.shape[0],z.shape[1],1,1).to(device)
        real=batch
        fake =upsampler(netG(z).detach().to(device))
    ssim=struct_similiarity(real,fake).cpu().detach().numpy()
    corr=pearson_coefficient(real,fake).cpu().detach().numpy()
    fsim=feature_similarity(real,fake).cpu().detach().numpy()
    
    structure=np.append(structure,ssim)
    pearson=np.append(pearson,corr)
    feature=np.append(feature,fsim)
    if i>10:
        break

s_mean=np.mean(structure)
p_mean=np.mean(pearson)
f_mean=np.mean(feature)

s_sd=np.std(structure)
p_sd=np.std(pearson)
f_sd=np.std(feature)


# In[58]:


print('Structural Similarity: ',s_mean,' +- ',s_sd)
print('Pearson Correlation: ',p_mean,' +- ',p_sd)
print('Feature Similarity: ',f_mean,' +- ',f_sd)


# In[ ]:


# # VISUALISE SOME OF THE OUTPUTS
# real_batch = next(iter(dataloader_test))
# with torch.no_grad():
#     k=28  # VARY K to display different images
#     z=PCA_transform(phi(real_batch[0]),pca_mat)
#     z=z[k].reshape(1,z.shape[1],1,1).to(device)
#     real=real_batch[0][k]
#     fake = netG(z).detach().to(device)
# real_img=vutils.make_grid(real, padding=2, normalize=True)
# fake_img=vutils.make_grid(fake, padding=2, normalize=True)
# plt.figure(0)
# plt.imshow(real_img.cpu().permute(1, 2, 0))
# plt.figure(1)
# plt.imshow(fake_img.cpu().permute(1, 2, 0))

