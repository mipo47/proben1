from proben1_reader import *
from proben_data import * # gates engine
import catboost

# choose dataset to train
DATASET = DATA_INFO.soybean

# gets same result on each run
np.random.seed(0)

# read data
data_name = DATASET.name
data_info = DATASET.value
data_path = 'proben1/' + data_name + '/' + data_name + '1.dt'
train, validation, test = read_proben1(data_path)
print("loaded data", data_name, train.length(), validation.length(), test.length())
print("input/output count", train.input_count(), train.output_count())

model = catboost.CatBoostClassifier(
    iterations=10000,
    verbose=100
)

y_train = np.argmax(train.outputs, axis=1)
y_val = np.argmax(validation.outputs, axis=1)

model.fit(
    train.inputs, y_train,
    eval_set=(validation.inputs, y_val),
)