import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim

temp = 1
hard = False
latent_dim = 30
categorical_dim = 10  # one-of-K vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


class VAE_gumbel(nn.Module):
    def __init__(self):
        super(VAE_gumbel, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))

    def decode(self, z):
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5))

    def forward(self, x):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

    def loss_function(self, recon_x, qy, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) / x.shape[0]

        log_ratio = torch.log(qy * categorical_dim + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean()

        return BCE + KLD

    def sample(self, device):
        M = 64 * latent_dim
        np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        
        sample = sample.to(device)

        sample = self.decode(sample).cpu()
        
        return sample.data.view(M // latent_dim, 1, 28, 28)
