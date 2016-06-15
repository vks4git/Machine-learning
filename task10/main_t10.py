import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def evaluate(vec, data):
    batch_size_ae, batch_size, factor, epocs_ae, epocs = int(vec[0]), int(vec[1]), vec[2], int(vec[3]), int(vec[4])
    layer1_count_ae = int(784 * factor)
    layer2_count_ae = int(layer1_count_ae * factor)
    layer3_count_ae = layer2_count_ae
    layer4_count_ae = layer1_count_ae

    w1 = tf.Variable(tf.random_normal([784, layer1_count_ae]))
    w2 = tf.Variable(tf.random_normal([layer1_count_ae, layer2_count_ae]))
    w3 = tf.Variable(tf.random_normal([layer2_count_ae, layer3_count_ae]))
    w4 = tf.Variable(tf.random_normal([layer3_count_ae, layer4_count_ae]))
    w_out_ae = tf.Variable(tf.random_normal([layer4_count_ae, 784]))

    bias_1 = tf.Variable(tf.random_normal([layer1_count_ae]))
    bias_2 = tf.Variable(tf.random_normal([layer2_count_ae]))
    bias_3 = tf.Variable(tf.random_normal([layer3_count_ae]))
    bias_4 = tf.Variable(tf.random_normal([layer4_count_ae]))
    bias_out_ae = tf.Variable(tf.random_normal([784]))

    X = tf.placeholder(tf.float32, [None, 784])
    y_t_i_ae = tf.placeholder(tf.float32, [None, 784])

    layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w1), bias_1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), bias_2))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w3), bias_3))
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, w4), bias_4))
    prediction_ae = tf.nn.relu(tf.add(tf.matmul(layer_4, w_out_ae), bias_out_ae))

    cross_entropy_ae = tf.reduce_mean(-tf.reduce_sum(y_t_i_ae * tf.log(prediction_ae), reduction_indices=[1]))
    train_step_ae = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy_ae)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epocs_ae):
        batch_xs, _ = data.train.next_batch(batch_size_ae)
        sess.run(train_step_ae, feed_dict={X: batch_xs, y_t_i_ae: batch_xs})

    # ******************************************************************************************************************

    w_out = tf.Variable(tf.random_normal([layer2_count_ae, 10]))
    bias_out = tf.Variable(tf.random_normal([10]))
    y_t_i = tf.placeholder(tf.float32, [None, 10])

    prediction = tf.matmul(layer_2, w_out) + bias_out

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_t_i))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epocs):
        batch_xs, batch_ys = data.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X: batch_xs, y_t_i: batch_ys})

    target = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_t_i, 1))
    score = tf.reduce_mean(tf.cast(target, "float"))
    return sess.run(score, feed_dict={X: data.test.images, y_t_i: data.test.labels})

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(evaluate([6151, 49, 0.728933, 9, 109], mnist))
