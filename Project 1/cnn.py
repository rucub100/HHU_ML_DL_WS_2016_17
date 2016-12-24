from __future__ import absolute_import
from __future__ import division
import json
import sys
import logging

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("data/", one_hot=True)


"""
Define layer size etc
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger('CNN')

config = None
with open("config.json", "r") as config_file:
    config = json.load(config_file)
if config == None:
    print "Config file 'config.json' not found"
    sys.exit(1)

LEARNING_RATE = config['learning_rate']
log.info("Learning reate: %2f" % LEARNING_RATE)
#use batches to shiffer through reasonable data patches
BATCH_SIZE = config['training_round_size']
log.info("Batch size: %d" % BATCH_SIZE)
#784=28*28 pixels in all mnist pics
MNIST_WIDTH = 28
MNIST_HEIGHT = 28
#pictures are divided into ten different classes
MNIST_CLASSES = 10
#number of pixels in one picture
MNIST_PIXELS = MNIST_WIDTH * MNIST_HEIGHT
#placing shape here is useful if you have that info, because tf will throw error if other pic sizes are flowing in! :)
x = tf.placeholder('float',[None, MNIST_PIXELS])
y = tf.placeholder('float')
y_ = tf.placeholder(tf.float32, [None, MNIST_CLASSES])
sess = tf.InteractiveSession()


def weight_variable(shape):
    """Creates a tf Variable of shape 'shape' with random elements"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a small bias of shape 'shape' with contants values of 0.1"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """Use a conv layer with weights W on zero padded input x and stride 1"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """Create a 2x2 pooling layer with strides 2, reducing the resolution"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_neural_network(x):
    # input data is embedded in 4d tensor
    current_size = 28
    current_depth = 1
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # x is 1x28*28 ( x = [34,43,5,6,0,0,7,5,...] )
    log.info("Create comutational graph...")
    for layer in config['layers']:
        type = layer['type']
        if type == "convolution":
            width = layer['size']
            height = width
            depth = layer['depth']
            w_conv = weight_variable([width, height, current_depth, depth])
            b_conv = bias_variable([depth])
            x_image = tf.nn.relu(conv2d(x_image, w_conv) + b_conv)
            current_depth = depth
        elif type == "pool":
            x_image = max_pool_2x2(x_image)
            current_size = current_size / 2

    #Fully connected layer
    log.info("Create fully connected layer...")
    W_fc1 = weight_variable([int(current_size * current_size * current_depth), 50])
    b_fc1 = bias_variable([50])
    h_pool2_flat = tf.reshape(x_image, [-1, int(current_size * current_size * current_depth)])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([50, MNIST_CLASSES])
    b_fc2 = bias_variable([MNIST_CLASSES])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    training_rounds = config['training_rounds']
    training_round_size = config['training_round_size']
    log.info("Training rounds: %d" % training_rounds)
    log.info("Training round size: %d" % training_round_size)
    log.info("Start training...")
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(training_rounds * training_round_size):
        batch = mnist.train.next_batch(BATCH_SIZE)
        if i%training_round_size == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
            log.info("step %d, TRAINING ACCURACY: %f"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    i += 1
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    log.info("step %d, TRAINING ACCURACY: %f"%(i, train_accuracy))

    log.info("Start testing...")
    log.info("TEST ACCURACY: %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/", one_hot=True)
    try:
        train_neural_network(x)
    except KeyboardInterrupt:
        print
