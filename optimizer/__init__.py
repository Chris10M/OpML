from torch import optim
from .adam_Hd import Adam_HD
from .sgd_Hd import SGD_HD

Optimizers = {
    'adam': optim.Adam,
    'adam_Hd' : Adam_HD
    #'sgd_Hd' : SGD_HD
}