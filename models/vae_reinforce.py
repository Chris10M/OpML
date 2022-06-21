import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import OneHotCategorical


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 20*256)
        self.fc3 = nn.Linear(20*256, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(h1).view(len(x), 20, 256), -1)
        return probs

    def decode(self, z):
        h3 = F.relu(self.fc3(z.view(len(z), 20*256)))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        # For convenience we use torch.distributions to sample and compute the values of interest for the distribution see (https://pytorch.org/docs/stable/distributions.html) for more details.
        probs = self.encode(x.view(-1, 784))
        m = OneHotCategorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        return self.decode(action), log_prob, entropy

    def loss_function(self, recon_x, log_prob, entropy, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduce=False).sum(-1)
        # We make the assumption that q(z1,..,zd|x) = q(z1|x)...q(zd|x)
        log_prob = log_prob.sum(-1)
        # The Reinforce loss is just log_prob*loss
        reinforce_loss = torch.sum(log_prob*BCE.detach())
        # If the prior on the latent is uniform then the KL is just the entropy of q(z|x)
        # We add reinforce_loss - reinforce_loss.detach() so we can backpropagate through the encoder with REINFORCE but it doesn't modify the loss.
        loss = BCE.sum() + reinforce_loss - reinforce_loss.detach() + entropy.sum()

        return loss

    def sample(self, device):
        with torch.no_grad():
            m = OneHotCategorical(torch.ones(256)/256.)
            sample = m.sample((64, 20))
            sample = sample.to(device)
            sample = self.decode(sample).cpu()

        return sample.view(64, 1, 28, 28)
