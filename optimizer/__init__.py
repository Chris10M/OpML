from torch import optim
from .adam_Hd import Adam_HD
from .sgd_Hd import SGD_HD
from .op_adam_lop_adam import op_Adam_lop_Adam
from .adamw import AdamW
from .adafactor import Adafactor
from .madgrad import MADGRAD

Optimizers = {
    #'adam': optim.Adam,
    #'adam_op_adam_hd' : op_Adam_lop_Adam
    #'adam_Hd' : Adam_HD
    #'sgd_Hd' : SGD_HD
    #'Adam_w' : AdamW
    #'Adam_R' : Adafactor
    'MADGRAD': MADGRAD
}