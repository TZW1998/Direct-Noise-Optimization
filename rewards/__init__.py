from .rewards_zoo import *

RFUNCTIONS = {
    "aesthetic" : aesthetic_loss_fn,
    "hps" : hps_loss_fn,
    "pick" : pick_loss_fn,
    "white" : white_loss_fn,
    "black" : black_loss_fn,
    "jpeg": jpeg_compressibility
}