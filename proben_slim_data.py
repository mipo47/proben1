from enum import Enum
from gates.gate_net import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
# example benchmark http://fann.sourceforge.net/report/node8.html

LossL2 = tf.losses.mean_squared_error
SoftmaxLoss = tf.losses.softmax_cross_entropy
Tanh = tf.nn.tanh
Relu = tf.nn.relu
Sigmoid = tf.nn.sigmoid
BatchNorm = slim.batch_norm

# mean error train/validation/test: 0.02009, 0.02783, 0.02898
class DATA_INFO(Enum):
    building = DataInfo({
        'scores': [0.00155932, 0.00675868, 0.00509490],
        'layers': [16],
        # 'batch_size': 64,
        'loss': LossL2
    })
    building2 = DataInfo({
        'name': 'building',
        'scores': [0.00217419, 0.00576708, 0.00356113],
        'layers': [64, 64],
        'batch_size': 64,
        'activation': Tanh,
        'dropout': 0.2,
        'loss': LossL2
    })

    gene = DataInfo({
        'scores': [0.02956121, 0.04121585, 0.05438812],
        'layers': [4, 2],
        'activation': lambda prev: Tanh(BatchNorm(prev)),
        'batch_size': 23,
        'loss': SoftmaxLoss
    })

    card = DataInfo({
        'scores': [0.08812010, 0.07805184, 0.09294949],
        'layers': [32],
        'batch_size': 25,
        'loss': SoftmaxLoss
    })

    mushroom = DataInfo({
        'scores': [1.1489584e-10, 8.2236368e-10, 6.7191914e-09],
        'layers': [32],
        'loss': SoftmaxLoss
    })

    soybean = DataInfo({
        'scores': [0.00161429, 0.00313735, 0.00559704],
        'activation': lambda prev: Tanh(BatchNorm(prev)),
        'layers': [16, 8],
        'batch_size': 17,
        'loss': SoftmaxLoss
    })

    thyroid = DataInfo({
        # 'scores': [0.0039941357, 0.0090197045, 0.0080988621],
        'scores': [0.00302554, 0.01083102, 0.00940251],
        'activation': Relu,
        'layers': [16, 8],
        'loss': SoftmaxLoss
    })
    thyroid2 = DataInfo({
        'name': 'thyroid',
        'scores': [0.00347402, 0.00668362, 0.00807445],
        'activation': tf.nn.elu,
        'layers': [128, 64],
        'batch_size': 1024,
        'dropout': 0.2,
        'loss': SoftmaxLoss
    })
