from sklearn.metrics import log_loss, accuracy_score

from proben1_reader import *
from proben_data import * # gates engine
import lightgbm as lgb

# choose dataset to train
DATASET = DATA_INFO.gene

# gets same result on each run
np.random.seed(0)

# read data
data_name = DATASET.name
data_info = DATASET.value
data_path = 'proben1/' + data_name + '/' + data_name + '1.dt'
train, validation, test = read_proben1(data_path)
print("loaded data", data_name, train.length(), validation.length(), test.length())
print("input/output count", train.input_count(), train.output_count())

model = lgb.LGBMClassifier(
    learning_rate=0.02,
    n_estimators=10000
)

y_train = np.argmax(train.outputs, axis=1)
y_val = np.argmax(validation.outputs, axis=1)

model.fit(
    train.inputs, y_train,
    eval_set=(validation.inputs, y_val),
    early_stopping_rounds=10
)

y_pred = model.predict_proba(validation.inputs)
loss = log_loss(validation.outputs, y_pred)

y_pred = model.predict(validation.inputs)
score = accuracy_score(y_val, y_pred)
print(model.__class__.__name__, "Score / Loss", score, loss)
