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
        'gates': [0.05037028887118512, 0.1132455569261696, 0.19223504907944622],
        'xgboost': 0.14167,
        'lightgbm': 0.139782,
        'catboost': 0.1113634,
        'tf_keras': 0.3247,
        'layers': [4, 2],
        'activation': lambda prev: Tanh(BatchNorm(prev)),
        'batch_size': 23,
        'loss': SoftmaxLoss
    })

    card = DataInfo({
        'scores': [0.090754673, 0.086547732, 0.094834417],
        'gates': [0.29706301481827446, 0.29554614028489656, 0.31602761911791427],
        'xgboost': 0.27388,
        'lightgbm': 0.284746,
        'catboost': 0.2789839,
        'tf_keras': 0.2659356337406732,
        'layers': [32],
        # 'batch_size': 25,
        'loss': SoftmaxLoss
    })

    mushroom = DataInfo({
        'scores': [1.1489584e-10, 8.2236368e-10, 6.7191914e-09],
        'xgboost': 0.00040,
        'lightgbm': 9.99462e-07,
        'catboost': 0.001159685464,
        'tf_keras': 7.043383647533186e-10,
        'layers': [32],
        'loss': SoftmaxLoss
    })

    soybean = DataInfo({
        'scores': [0.0017386284, 0.0033119305, 0.0058070654],
        'gates': [0.05037028887118512, 0.1132455569261696, 0.19223504907944622],
        'xgboost': 0.14909,
        'lightgbm': 0.121413,
        'catboost': 0.1525098,
        'tf_keras': 0.11372763908240803,
        'activation': Relu,
        'layers': [16, 8],
        'batch_size': 17,
        'loss': SoftmaxLoss
    })

    thyroid = DataInfo({
        'scores': [0.0039941357, 0.0090197045, 0.0080988621],
        'gates': [0.022757653130425347, 0.06294529385036893, 0.06159390343560113],
        'xgboost': 0.01555,
        'lightgbm': 0.0141798,
        'catboost': 0.0165330,
        'tf_keras': 0.052, # 0.008834491454892688
        'activation': Relu,
        'layers': [16, 8],
        'loss': SoftmaxLoss
    })
