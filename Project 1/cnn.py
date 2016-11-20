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

# # Create the model
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.matmul(x, W) + b

# # Define loss and optimizer
# y_ = tf.placeholder(tf.float32, [None, 10])

# # The raw formulation of cross-entropy,
# #
# #	 tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
# #																 reduction_indices=[1]))
# #
# # can be numerically unstable.
# #
# # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# # outputs of 'y', and then average across the batch.
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# sess = tf.InteractiveSession()
# # Train
# tf.initialize_all_variables().run()
# for _ in range(1000):
# 	batch_xs, batch_ys = mnist.train.next_batch(100)
# 	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# # Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))



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
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	print type(prediction)
	print "prediction:", prediction
	print "y:", y
	print "cost:", cost
	print "-"*50
	# optional param learning rate, default = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#cycles of feed forward + backprop
	hm_epochs = 15
	#y = tf.placeholder('float')
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				# tf is very high level, lots of helper functions!!
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run(
					[optimizer, cost],
					feed_dict = {x:epoch_x, y:epoch_y}
				)
				epoch_loss += c
			print 'Epoch ',epoch,' completed out of ',hm_epochs, 'loss: ', epoch_loss

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})


if __name__ == "__main__":
	train_neural_network(x)




