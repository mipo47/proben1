import pprint

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from proben1_reader import *
from proben_data import * # gates engine

# choose dataset to train
DATASET = DATA_INFO.thyroid

# gets same result on each run
np.random.seed(0)

# read data
data_name = DATASET.name
data_info = DATASET.value
data_path = 'proben1/' + data_name + '/' + data_name + '1.dt'
train, validation, test = read_proben1(data_path)
print("loaded data", data_name, train.length(), validation.length(), test.length())
print("input/output count", train.input_count(), train.output_count())

models = [
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

y_train = np.argmax(train.outputs, axis=1)
y_val = np.argmax(validation.outputs, axis=1)

pp = pprint.PrettyPrinter(indent=4)
results = {}

for model in models:
    model.fit(train.inputs, y_train)

    y_pred = model.predict_proba(validation.inputs)
    loss = log_loss(validation.outputs, y_pred)

    score = model.score(validation.inputs, y_val)
    print(model.__class__.__name__, "Score / Loss", score, loss)

    results[model.__class__.__name__] = loss

print()
pp.pprint(results)
