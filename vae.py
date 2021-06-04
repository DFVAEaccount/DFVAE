import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy
from torchvision.utils import save_image
import torch.optim as optim
import os
from config import attachlabel,fineTuneFedModel,weights_init,parametersAvg

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
spare_device = torch.device("cpu")
class Resize(nn.Module):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)

class gendata(Dataset):
    def __init__(self,gen_imgs,targets):
        imgs = []
        for index in range(len(gen_imgs)):
            imgs.append((gen_imgs[index], targets[index]))
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        pic, label = self.imgs[index]
        pic = pic.to(spare_device)
        label = label.to(spare_device)
        return pic.detach(), label.detach()

def loss_function(args, recon_x, x, mu, logvar, z, sensitivedim, sensitiveattr, hyperparameterR):

    """
    :param recon_x: generated image
    :param x: original image
    :param mu: latent mean of z
    :param logvar: latent log variance of z
    :param alignreward: reward for sentitive attributes alignment
    :param hyperparameterR: a tradeoff hyperparameter for accuracy and alignReward
    """

    #attrnum = len(sensitiveattr[0])
    Criterion = nn.MSELoss(reduction = 'sum')
    reConstructionLoss = Criterion(recon_x, x)
    KLDivergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu**2)
    alignReward = Criterion(sensitivedim, sensitiveattr)
    return args.reLossFactor * reConstructionLoss +  hyperparameterR * (KLDivergence +  alignReward)

class VAE_FC(nn.Module):
    def __init__(self,imshape_1,impshape_2,in_chan = 3):
        super(VAE_FC, self).__init__()
        self.encoder = nn.Sequential(
            Resize((-1,imshape_1 * impshape_2 * in_chan)),
            nn.Linear(imshape_1 * impshape_2 * in_chan, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256)
        )
        self.mean = nn.Linear(256, 30)
        self.logvar = nn.Linear(256, 30)
        self.decoder = nn.Sequential(
            nn.Linear(30, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048,imshape_1 * impshape_2 * in_chan),
            Resize((-1,in_chan,imshape_1,impshape_2))
        )
        self.extramlp = nn.Sequential(
            nn.Linear(5,10),
            nn.Linear(10,5),
            nn.Linear(5,1)
        )
    def encode(self, x):
        h1 = self.encoder(x)
        return self.mean(h1), self.logvar(h1)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        z = torch.randn(std.size()).to(device) * std + mu
        return z

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, inputz = None):
        if inputz != None:
            return self.decode(inputz)
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)

        sensitivedim = self.extramlp(z[:,:5])
        output = self.decode(z)
        return output, mu, logvar, z, sensitivedim

class VAE(nn.Module):
    def __init__(self,imshape_1,imshape_2,in_chan = 3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chan, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            Resize((-1, 64 * imshape_1 * imshape_2 // 256)),
            nn.Linear(64 * imshape_1 * imshape_2 // 256, 256)
        )
        self.mean = nn.Linear(256,200)
        self.logvar = nn.Linear(256,200)

        self.decoder = nn.Sequential(
            nn.Linear(200, 256),
            nn.ReLU(True),
            nn.Linear(256, 64 * imshape_1 * imshape_2 // 256),
            nn.ReLU(True),
            Resize((-1, 64, imshape_1 // 16, imshape_2 // 16)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_chan, 4, 2, 1)
        )
    def encode(self, x):
        h1 = self.encoder(x)
        return self.mean(h1), self.logvar(h1)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        z = torch.randn(std.size()).to(device) * std + mu
        return z

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)

        return self.decode(z), mu, logvar, z

def adversary(args,contriData, fedModel, rSet = None):
    vae = VAE_FC(args.imshape_1, args.imshape_2, args.in_chan)
    if os.path.exists('./vaefc_' + args.name + '.pth'):
        vae.load_state_dict(torch.load('./vaefc_' + args.name + '.pth'))
    else:
        weights_init(vae)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    '''
    This is an optimizer parameters loader. No need to delete.  
    if os.path.exists('./optimizer' + args.name + '.pth'):
        optimizer.load_state_dict(torch.load('./optimizer' + args.name + '.pth'))
    '''
    if not os.path.exists('FAKE-' + args.name):
       os.makedirs('FAKE-' + args.name)

    gen_dataset = []

    for index in range(len(contriData)):
        contriLoader = DataLoader(contriData[index], batch_size = 256, shuffle = False)

        # This is training process
        for epoch in range(args.encodingEpochs):

            for batch_idx, (inputs, targets) in enumerate(contriLoader):
                inputs, targets = inputs.to(device), targets.to(device)
                real_imgs = inputs
                sensitiveLabels = attachlabel(args,targets).to(device)
                # Train Encoder
                optimizer.zero_grad()
                gen_imgs, mu, logvar, z, sensitivedim = vae(real_imgs)
                loss = loss_function(args, gen_imgs, real_imgs, mu, logvar,
                                     z, sensitivedim, sensitiveLabels,rSet[index])
                #print(loss.item())
                loss.backward()
                optimizer.step()

                # Save generated images for every epoch
                '''
                if epoch == args.encodingEpochs - 1:
                    print("saved generated images")
                    save_image(gen_imgs, './FAKE-{}/gen_images_{}.png'.format(args.name, index))
                '''
        print('Adversarial for one party is ready')
        # next is generating process
        for batch_idx, (inputs, targets) in enumerate(contriLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            real_imgs = inputs
            # only decoder is enough
            gen_imgs, mu, logvar, z, sensitivedim = vae(real_imgs)
            inputz = torch.cat([torch.randn(len(z),5).to(device),z[:,5:]],1)
            fakeImages = vae(real_imgs, inputz = inputz)

            fakeImages, targets = fakeImages.to(spare_device),targets.to(spare_device)
            if len(gen_dataset) < index + 1:
                gen_dataset.append(gendata(fakeImages, targets))
            else:
                gen_dataset[index] += gendata(fakeImages, targets)

    #torch.save(optimizer.state_dict(),'./optimizer' + args.name + '.pth')
    fedStateDict = deepcopy(fedModel.state_dict())
    fineTuneFedModel(fedModel,contriData,args)
    fineTuneFedModel(fedModel,gen_dataset,args)
    parametersAvg(fedStateDict, fedModel, 0.9)
    torch.save(vae.to(spare_device).state_dict(), './vaefc_' + args.name + '.pth')
    return fedModel

