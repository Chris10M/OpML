import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from args import args, kwargs

from optimizer import Optimizers
from models.vae_discrete import VAE as VAE_discrete
from models.vae_gumel_softmax import VAE_gumbel
from models.vae_reinforce import VAE as VAE_reinforce


MODELS = {
    'discrete': VAE_discrete(),
    'gumbel': VAE_gumbel(),
    'reinforce': VAE_reinforce(),
}   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device) 
        optimizer.zero_grad()
        outs = model(data)
        loss = model.loss_function(*outs, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(model, test_loader):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            outs = model(data)
            test_loss += model.loss_function(*outs, data).item()

    test_loss /= len(test_loader.dataset)
    
    return test_loss


def run(model, optimizer, train_loader, test_loader):   
    for epoch in range(1, args.epochs + 1):    
        train(model, optimizer, train_loader, epoch)
        test_loss = test(model, test_loader)

        samples = model.sample(device)
            # save_image(sample.view(64, 1, 28, 28),
            #         'resultsDiscrete/sample_' + str(epoch) + '.png')
        print(test_loss, samples.shape)


def main():
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

    for model_name, model in MODELS.items():
        for optim_name, optimizer in Optimizers.items():
            print(model_name, optim_name)
            run(model.to(device), optimizer(params=model.parameters()), train_loader, test_loader)
            
        # print(optim_name)
        # model = VAE().to(device)
        # optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # train(model, optimizer, train_loader, 0)    
        # print('====> Test set loss: {:.4f}'.format(test_loss))
        # if i == 0:r
        #     n = min(data.size(0), 8)
        #     comparison = torch.cat([data[:n],
        #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        #     save_image(comparison.cpu(),
        #              'resultsDiscrete/reconstruction_' + str(epoch) + '.png', nrow=n)


if __name__ == '__main__':
    main()
