from gates.proben1 import *
from proben_slim_data import *

# choose dataset to train
DATASET = DATA_INFO.card
MAX_STEPS = 100000
MAX_EXPLORE_STEPS = 401
REWIND_COUNT = 3

# read data
data_name = DATASET.name
data_info = DATASET.value
if 'name' in data_info:
    data_name = data_info.name
data_path = 'proben1/' + data_name + '/' + data_name + '1.dt'
train, validation, test = read_proben1(data_path)
print("loaded data", data_name, train.length(), validation.length(), test.length())
print("input/output count", train.input_count(), train.output_count())


def get_input_fn(data, shuffle=False, num_epochs=1):
    one_hot = np.argmax(data.outputs, axis=-1)
    return tf.estimator.inputs.numpy_input_fn(
        {"x": data.inputs},
        one_hot,
        num_epochs=num_epochs,
        shuffle=shuffle)

with tf.device("cpu:0"):
    train_input_fn = get_input_fn(train, shuffle=True, num_epochs=None)
    validation_input_fn = get_input_fn(validation)
    test_input_fn = get_input_fn(test)

    feature_columns = [tf.feature_column.numeric_column("x", shape=[train.input_count()])]

    classifier = tf.estimator.DNNClassifier(
        data_info.layers,
        feature_columns,
        optimizer=tf.train.AdamOptimizer(0.01, 0.001, 0.9),
        activation_fn=data_info.activation if 'activation' in data_info else tf.nn.tanh,
        dropout=data_info.dropout if 'dropout' in data_info else None,
        n_classes=train.output_count())

    for i in range(100):
        classifier.train(train_input_fn, max_steps=10)
        scores = classifier.evaluate(validation_input_fn)
        print(scores)