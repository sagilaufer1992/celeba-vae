from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import operator
import os


parser = argparse.ArgumentParser(description='VAE CELEBA Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-features', type=int, default=2048, metavar='N',
                    help='how many mus and sigmas to create')
parser.add_argument('--num-images', type=int, default=0, metavar='N',
                    help='how many images to take from celeba')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

num_features = args.num_features
num_classes = 4


class my_celeba(datasets.CelebA):
    def __len__(self):
        return args.num_images if args.num_images > 0 else super().__len__()


my_dataset = my_celeba('.', transform=transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))

file1 = open(os.path.join('.', 'celeba', 'list_attr_celeba.txt'), 'r')
Lines = file1.readlines()
classes = Lines[1].split(' ')[0:-1]
en = {i: x for i, x in enumerate(classes)}
selector = torch.zeros(40, dtype=torch.uint8)
selector[9] = 1
selector[15] = 1
selector[20] = 1
selector[31] = 1

print(operator.itemgetter(9, 15, 20, 31)(classes))

train_loader = torch.utils.data.DataLoader(
    dataset=my_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        nc = 3
        ndf = 64
        ngf = 64

        self.main_encode = nn.Sequential(
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
        )

        self.mu_conv = nn.Sequential(
            nn.Conv2d(ndf * 8, num_features, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )
        self.std_conv = nn.Sequential(
            nn.Conv2d(ndf * 8, num_features, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        self.cls_conv = nn.Sequential(
            nn.Conv2d(ndf * 8, num_classes, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        self.pre_decode = nn.Sequential(
            nn.Linear(num_features + num_classes, num_features),
            nn.Tanh())

        self.main_decode = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(num_features, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def encode(self, x):
        h1 = self.main_encode(x)
        return self.mu_conv(h1), self.std_conv(h1), self.cls_conv(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, cls):
        concatenated = torch.cat((z, cls), dim=1)
        size = concatenated.size()
        pre = self.pre_decode(concatenated.view(size[0], -1))
        h3 = self.main_decode(pre.view(size[0], -1, 1, 1))
        return torch.tanh(h3)

    def forward(self, x, target):
        mu, logvar, cls = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, target), mu, logvar, cls


model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, cls, target):
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    target_loss = F.mse_loss(cls, target)

    return BCE + KLD + target_loss, BCE, KLD + target_loss


def train(epoch):
    model.train()
    train_loss = 0
    bce_loss = 0
    kld_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # if batch_idx == 0:
        #     new_data = data * 0.5 + 0.5
        #     save_image()
        target = _[:, selector].to(device).view(data.size()[0], num_classes, 1, 1).float()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, cls = model(data, target)
        # recon_batch = recon_batch * 0.5 + 0.5
        # data = data * 0.5 + 0.5
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, cls, target)

        loss.backward()

        train_loss += loss.item()
        bce_loss += bce.item()
        kld_loss += kld.item()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    with open('bce_loss.txt', 'a') as f:
        f.write('{:.4f}'.format(bce_loss / len(train_loader.dataset)) + "\n")
    with open('kld_loss.txt', 'a') as f:
        f.write('{:.4f}'.format(kld_loss / len(train_loader.dataset)) + "\n")


def reproduce_hw3():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        # test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, num_features, 1, 1).to(device)
            rand_cls = torch.randn(64, num_classes, 1, 1).sigmoid().round().to(device)

            sample = model.decode(sample, rand_cls)
            sample = (sample * 0.5) + 0.5
            save_image(sample.view(64, 3, 64, 64),
                       'results/sample_' + str(epoch) + '.png')

        torch.save(model.state_dict(), "models/vae.pth")


if __name__ == "__main__":
    reproduce_hw3()
