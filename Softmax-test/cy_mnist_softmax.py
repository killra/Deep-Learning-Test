from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

FLAGS = None

def distribution():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	count = {}
	for onerecord in mnist.train.labels:
		num = 0
		for x in range(10):
			if onerecord[x] != 0.0:
				num = x
				break

		if num in count:
			count[num] += 1
		else:
			count[num] = 1

	key = []
	value = []
	for k, v in count.items():
		key.append(k)
		value.append(v)
		print 'key: %d,  count: %d' % (k, v)

	y_pos = np.arange(len(key))
	plt.bar(y_pos, value, align='center', alpha=0.5)
	plt.xticks(y_pos, key)
	plt.ylabel('Count')
	plt.title('label distribution')
	plt.show()


def main(_):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x, W) + b

	y_ = tf.placeholder(tf.float32, [None, 10])


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	print(sess.run(accuracy, feed_dict={x: mnist.test.images,
										y_: mnist.test.labels}))


if __name__ == '__main__':
	#tf.app.run(main=main, argv=None)
	distribution()