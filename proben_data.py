from enum import Enum
from gates.gate_net import *
# example benchmark http://fann.sourceforge.net/report/node8.html


# mean error train/validation/test: 0.02009, 0.02783, 0.02898
class DATA_INFO(Enum):
    building = DataInfo({
        'scores': [0.0013624951, 0.0068991315, 0.006376252],
        'layers': [16],
        # 'batch_size': 64,
        'loss': LossL2
    })

    gene = DataInfo({
        'scores': [0.02014219, 0.054264467, 0.054950446],
        'layers': [4, 2],
        'activation': lambda prev: Tanh(BatchNorm(prev)),
        'batch_size': 23,
        'loss': SoftmaxLoss
    })

    card = DataInfo({
        'scores': [0.090754673, 0.086547732, 0.094834417],
        'layers': [32],
        # 'batch_size': 25,
        'loss': SoftmaxLoss
    })

    mushroom = DataInfo({
        'scores': [1.1489584e-10, 8.2236368e-10, 6.7191914e-09],
        'layers': [32],
        'loss': SoftmaxLoss
    })

    soybean = DataInfo({
        'scores': [0.0017386284, 0.0033119305, 0.0058070654],
        'activation': Relu,
        'layers': [16, 8],
        'batch_size': 17,
        'loss': SoftmaxLoss
    })

    thyroid = DataInfo({
        'scores': [0.0039941357, 0.0090197045, 0.0080988621],
        'activation': Relu,
        'layers': [16, 8],
        'loss': SoftmaxLoss
    })
