from torch import optim
from functools import *
from .adam_Hd import Adam_HD
from .op_adam_lop_adam import op_Adam_lop_Adam
from .adamw import AdamW
from .AdamP import AdamP
from .adafactor import Adafactor
from .madgrad import MADGRAD
from .nvnovograd import NvNovoGrad
from .AdamS import Adams
from .Nadam import Nadam

Optimizers = {
    #'adam': optim.Adam,
    #'SGD': partial(optim.SGD, lr=1e-3)
    #'SGD_M': partial(optim.SGD, lr=1e-3, momentum=0.9)
    #'adam_op_adam_hd' : op_Adam_lop_Adam,
    #'adam_Hd' : Adam_HD,
    #'Adam_w' : AdamW,
    'Adam_w' : AdamP,
    #'Adafactor' : Adafactor,
    #'MADGRAD': MADGRAD,
    #'NvNovoGrad' : NvNovoGrad,
    #'AdaBelief' : Adams,
    #'Nadam': Nadam
}