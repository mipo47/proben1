from proben1_reader import *
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


optimizer = tf.train.AdamOptimizer(0.01, 0.001, 0.9)
is_training = tf.Variable(True, dtype=tf.bool)


def create_model(x, y):
    # default activation function for hidden layers
    activation = data_info.activation if 'activation' in data_info else tf.nn.tanh
    net = x
    with tf.name_scope("hidden"):
        for i, layer_size in enumerate(data_info.layers):
            net = slim.fully_connected(net, layer_size, activation_fn=activation, scope="fc_" + str(i))
            if 'dropout' in data_info:
                net = tf.layers.dropout(net, data_info.dropout, training=is_training)

    # last/output layer
    output_count = data_info.output_count if 'output_count' in data_info else train.output_count()
    is_classification = data_info.loss == SoftmaxLoss
    net = slim.fully_connected(net,
                               output_count,
                               activation_fn=None if is_classification else Sigmoid,
                               scope="predictions")

    loss = data_info.loss(y, net)

    # always display L2 loss score, even for softmax classification
    display_loss = loss \
        if data_info.loss == LossL2 \
        else tf.losses.mean_squared_error(y, tf.nn.softmax(net))

    train_op = slim.learning.create_train_op(loss, optimizer)

    return train_op, display_loss


with tf.device("cpu:0"):
    x = tf.placeholder(tf.float32, (None, validation.input_count()), "features")
    y = tf.placeholder(tf.float32, (None, validation.output_count()), "labels")
    model, l2_loss= create_model(x, y)

    batch_size = data_info.batch_size if 'batch_size' in data_info else train.length()

    sess = tf.Session()
    with sess:
        sess.run(tf.global_variables_initializer())

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        best_weights = []
        for var in weights:
            initial_value = sess.run(var)
            best_weights.append(tf.Variable(initial_value))

        updates = []
        for i, var in enumerate(weights):
            updates.append(best_weights[i].assign(var))
        update_best = tf.group(*updates, name='update_best_model')

        updates = []
        for i, best_var in enumerate(best_weights):
            updates.append(weights[i].assign(best_var))
        load_best = tf.group(*updates, name='load_best_model')

        data_loss = lambda data: sess.run(l2_loss, feed_dict={
            x: data.inputs,
            y: data.outputs,
            is_training: False
        })

        attempt = 0
        best_validation = 1e10
        best_step = 0
        for step in range(MAX_STEPS):
            features, labels = train.get_batch(batch_size)
            train_loss, train_l2 = sess.run([model, l2_loss], feed_dict={
                x: features,
                y: labels,
                is_training: True
            })

            if step % 10 == 0:
                val_loss = data_loss(validation)

                if step % 100 == 0 or val_loss < best_validation:
                    print(step, train_l2, val_loss, "Record" if val_loss < best_validation else "")

                if val_loss < best_validation:
                    sess.run(update_best)
                    best_validation = val_loss
                    best_step = step
                elif step > best_step + MAX_EXPLORE_STEPS:
                    attempt += 1
                    if attempt > REWIND_COUNT:
                        print(step, "validation", val_loss)
                        break
                    else:
                        print("Loading best weights", attempt)
                        best_step = step
                        sess.run(load_best)

        sess.run(load_best)
        train_loss = data_loss(train)
        val_loss = data_loss(validation)
        test_loss = data_loss(test)
        print("'scores': [{:.8f}, {:.8f}, {:.8f}],".format(train_loss, val_loss, test_loss))