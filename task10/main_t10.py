import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

layer1_count_ae = 784
layer2_count_ae = 392
batch_size = 7500

w1 = tf.Variable(tf.random_normal([784, layer1_count_ae]))
w2 = tf.Variable(tf.random_normal([layer1_count_ae, layer2_count_ae]))
w_out_ae = tf.Variable(tf.random_normal([layer2_count_ae, 784]))

bias_1 = tf.Variable(tf.random_normal([layer1_count_ae]))
bias_2 = tf.Variable(tf.random_normal([layer2_count_ae]))
bias_out_ae = tf.Variable(tf.random_normal([784]))

X = tf.placeholder(tf.float32, [None, 784])
y_t_i_ae = tf.placeholder(tf.float32, [None, 784])

layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w1), bias_1))
layer_2_ae = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), bias_2))
prediction_ae = tf.nn.relu(tf.add(tf.matmul(layer_2_ae, w_out_ae), bias_out_ae))


cross_entropy_ae = tf.reduce_mean(-tf.reduce_sum(y_t_i_ae * tf.log(prediction_ae), reduction_indices=[1]))
train_step_ae = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cross_entropy_ae)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

epocs_ae = 200

for i in range(epocs_ae):
    if i % (epocs_ae // 10) == 0:
        print("Learning AE: %i%%" % (100 * i // epocs_ae))
    batch_xs, _ = mnist.train.next_batch(batch_size)
    sess.run(train_step_ae, feed_dict={X: batch_xs, y_t_i_ae: batch_xs})

print("Done.")

# *********************************************************************************************************************

batch_size = 200
epocs = 5000
w_out = tf.Variable(tf.random_normal([layer1_count_ae, 10]))
bias_out = tf.Variable(tf.random_normal([10]))
y_t_i = tf.placeholder(tf.float32, [None, 10])

prediction = tf.matmul(layer_1, w_out) + bias_out

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_t_i))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(epocs):
    if i % (epocs // 10) == 0:
        print("Learning: %i%%" % (100 * i // epocs))
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_xs, y_t_i: batch_ys})

target = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_t_i, 1))
score = tf.reduce_mean(tf.cast(target, "float"))
print("Accuracy : %s" % sess.run(score, feed_dict={X: mnist.test.images, y_t_i: mnist.test.labels}))
