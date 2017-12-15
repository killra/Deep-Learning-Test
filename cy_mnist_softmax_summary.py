from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

FLAGS = None


'''
To show the labal distribution of traing data by histogram.
'''
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


'''
Softmax code
'''
def main(_):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	with tf.name_scope('input_layer'):
		x = tf.placeholder(tf.float32, [None, 784], name='input_data')
		W = tf.Variable(tf.zeros([784, 10]), name='weights')
		b = tf.Variable(tf.zeros([10]), name='biases')
		y = tf.matmul(x, W) + b
		y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

		variable_summaries(W, 'Weights')
		variable_summaries(b, 'bias')

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	
	with tf.name_scope('traing'):
		train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):	
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy',accuracy)
	
	merged = tf.summary.merge_all()


	sess = tf.InteractiveSession()	
	train_writer = tf.summary.FileWriter('Output/train', sess.graph)
	test_writer = tf.summary.FileWriter('Output/test', sess.graph)
	tf.global_variables_initializer().run()



	saver = tf.train.Saver()
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		#sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
		train_writer.add_summary(summary, i)


		summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images,
										y_: mnist.test.labels})
		test_writer.add_summary(summary, i)
		print('accuracy at step %s: %s' % (i, acc))

	train_writer.close()
	test_writer.close()


def variable_summaries(var, name):
	with tf.name_scope('summaries_' + name):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)

		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)

		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)



if __name__ == '__main__':
	tf.app.run(main=main, argv=None)
	#distribution()