from gates.proben1 import *
from gates.trainer import train_model
from proben_data import * # gates engine
# from proben_data_tf import * # tensorflow engine

# choose dataset to train
DATASET = DATA_INFO.building

# gets same result on each run
np.random.seed(0)

# read data
data_name = DATASET.name
data_info = DATASET.value
data_path = 'proben1/' + data_name + '/' + data_name + '1.dt'
train, validation, test = read_proben1(data_path)
print("loaded data", data_name, train.length(), validation.length(), test.length())
print("input/output count", train.input_count(), train.output_count())

train_model(data_info, train, validation, test, display_L2_loss=True)