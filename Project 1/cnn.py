from __future__ import absolute_import
from __future__ import division

import argparse

import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)


"""
Define layer size etc
"""

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
#use batches to shiffer through reasonable data patches
batch_size = 100
#784=28*28 pixels in all mnist pics
mnist_pixels = 784
#placing shape here is useful if you have that info, because tf will throw error if other pic sizes are flowing in! :)
x = tf.placeholder('float',[None, mnist_pixels])
y = tf.placeholder('float')
y_ = tf.placeholder(tf.float32, [None, 10])
sess = tf.InteractiveSession()

# 
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

def neural_network_model_2(x):
	return y_conv


"""
Define Neural Network
"""
def neural_network_model(data):
	#(input_data * weight) + biases
	hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([mnist_pixels,n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	print hidden_1_layer
	hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	#(input_data * weight) + biases ; rectified linear activation fct
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']) , output_layer['biases'])
	# we now have basically set up the computation graph, now we need train the model... 
	return output

def train_neural_network(x):
	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	W_fc1 = weight_variable([28 * 28 * 32, 50])
	b_fc1 = bias_variable([50])
	h_conv1_flat = tf.reshape(h_conv1, [-1, 28*28*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([50, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
					x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == "__main__":
	train_neural_network(x)




