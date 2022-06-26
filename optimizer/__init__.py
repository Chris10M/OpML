from torch import optim
from functools import *
from qhoptim.pyt import QHM, QHAdam
from .adam_Hd import Adam_HD
from .op_adam_lop_adam import op_Adam_lop_Adam
from .AdamP import AdamP
from .adafactor import Adafactor
from .madgrad import MADGRAD
from .nvnovograd import NvNovoGrad
from .AdamS import Adams
from .Nadam import Nadam

Optimizers = {
    'adam': optim.Adam,
    'Adam_w' : partial(optim.AdamW, lr=1e-3),
    'QHAdam' : partial(QHAdam, lr=1e-3, nus=(0.7, 1.0), betas=(0.995, 0.999)),
    'Adam_P' : AdamP,
    'Adafactor' : Adafactor,
    'MADGRAD': MADGRAD,
    'NvNovoGrad' : NvNovoGrad,
    'AdamS' : Adams,
    'Nadam': Nadam,
    #Hypergradient based optimizers
    'adam_Hd' : Adam_HD,
    'adam_op_adam_hd' : op_Adam_lop_Adam
}