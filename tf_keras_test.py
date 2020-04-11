import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from proben1_reader import *
from proben_data import * # gates engine
import tensorflow as tf

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

# layers = [
#     tf.keras.layers.Dense(size, activation='relu')
#     for size in data_info.layers
# ]
# layers.append(tf.keras.layers.Dense(train.output_count(), activation='softmax'))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(train.output_count(), activation='softmax')
])
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Nadam()
)

best_val_weights = None
best_val_loss = 99999
train_losses = []
val_losses = []

for t in range(5):
    no_improvement = 0
    for i in range(1000):
        history = model.fit(
            train.inputs, train.outputs,
            epochs=10,
            batch_size=data_info.batch_size if 'batch_size' in data_info else 1024,
            shuffle=True,
            validation_data=(validation.inputs, validation.outputs),
            validation_freq=10,
            verbose=0
        )
        train_losses.extend(history.history['loss'])
        if validation is None:
            continue

        val_losses.extend(history.history['val_loss'])

        val_loss = val_losses[-1]
        if val_loss < best_val_loss:
            best_val_weights = model.get_weights()
            best_val_loss = val_loss
            print('losses', train_losses[-1], best_val_loss)
            best_train_loss = train_losses[-1]
            no_improvement = 0
            improved = True
        else:
            no_improvement += 1
            if no_improvement >= 10:
                print(f'stop after {i + 1} rounds')
                break

    model.set_weights(best_val_weights)
    print('Try again')
