import numpy as np
import os
import csv

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from args import args, kwargs

from optimizer import Optimizers
from models.vae_discrete import VAE as VAE_discrete
from models.vae_gumel_softmax import VAE_gumbel
from models.vae_reinforce import VAE as VAE_reinforce


MODELS = {
    'discrete': VAE_discrete,
    'gumbel': VAE_gumbel,
    'reinforce': VAE_reinforce,
}   


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, train_loader, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device) 
        optimizer.zero_grad()
        outs = model(data, batch_idx)
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
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            outs = model(data, batch_idx)
            test_loss += model.loss_function(*outs, data).item()

    test_loss /= len(test_loader.dataset)
    
    return test_loss


def run(model, optimizer, train_loader, test_loader): 
    metrics = dict()  
    for epoch in range(1, args.epochs + 1):    
        train(model, optimizer, train_loader, epoch)
        test_loss = test(model, test_loader)
        samples = model.sample(device)

        metrics[epoch] = {
            'loss': test_loss,
            'samples': samples,
        }

    return metrics


def main():
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

    for model_name, model in MODELS.items():
        os.makedirs(f'results/{model_name}', exist_ok=True)

        for optim_name, optimizer in Optimizers.items():
            os.makedirs(f'results/{model_name}/{optim_name}/image', exist_ok=True)

            model_instance = model()
            metrics = run(model_instance.to(device), optimizer(params=model_instance.parameters()), train_loader, test_loader)

            losses = list()
            for epoch, item in metrics.items():
                losses.append([epoch, item['loss']])
                save_image(item['samples'], f'results/{model_name}/{optim_name}/image/{epoch}.png', nrow=8)

            with open(f'results/{model_name}/{optim_name}/test_loss.csv', 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(losses)


if __name__ == '__main__':
    main()