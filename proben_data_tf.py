from enum import Enum
from gates.tf_to_gates import *
from gates.data_info import *

# example benchmark http://fann.sourceforge.net/report/node8.html

class DATA_INFO(Enum):
    building = {
        'scores': [0.0012720634, 0.0080860108, 0.0062029539],
        'layers': [16],
        'batch_size': 25,
        'loss': LossL2
    }

    gene = {
        'scores': [0.19268422, 0.20313385, 0.19920653],
        'layers': [4, 2],
        'batch_size': 25,
        'loss': SoftmaxLoss
    }

    card = DataInfo({
        'scores': [0.023790944, 0.1093806, 0.13934357],
        'layers': [32],
        'loss': SoftmaxLoss
    })

    mushroom = {
        'scores': [],
        'layers': [32],
        'loss': SoftmaxLoss
    }

    soybean = {
        'scores': [],
        'activation': Relu,
        'layers': [16, 8],
        'batch_size': 17,
        'loss': SoftmaxLoss
    }

    thyroid = {
        'scores': [],
        'activation': Relu,
        'layers': [16, 8],
        'loss': SoftmaxLoss
    }
