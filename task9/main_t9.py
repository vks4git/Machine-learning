import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

layer1_count = 420
layer2_count = 420
batch_size = 228

w1 = tf.Variable(tf.random_normal([784, layer1_count]))
w2 = tf.Variable(tf.random_normal([layer1_count, layer2_count]))
w_out = tf.Variable(tf.random_normal([layer1_count, 10]))

bias_1 = tf.Variable(tf.random_normal([layer1_count]))
bias_2 = tf.Variable(tf.random_normal([layer2_count]))
bias_out = tf.Variable(tf.random_normal([10]))

X = tf.placeholder(tf.float32, [None, 784])
y_t_i = tf.placeholder(tf.float32, [None, 10])

layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w1), bias_1))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), bias_2))
prediction = tf.matmul(layer_1, w_out) + bias_out

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_t_i))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

epocs = 5000

for i in range(epocs):
    if i % (epocs // 10) == 0:
        print("Learning: %i%%" % (100 * i // epocs))
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_xs, y_t_i: batch_ys})

target = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_t_i, 1))
score = tf.reduce_mean(tf.cast(target, "float"))
print("Accuracy : %s" % sess.run(score, feed_dict={X: mnist.test.images, y_t_i: mnist.test.labels}))
