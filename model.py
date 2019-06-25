# the set of CNN model architectures

import tensorflow as tf


def show_num_parameters():
	total_parameters = 0
	for variable in tf.trainable_variables():
	    # shape is an array of tf.Dimension
	    shape = variable.get_shape()
	    # print(shape)
	    # print(len(shape))
	    variable_parametes = 1
	    for dim in shape:
	        # print(dim)
	        variable_parametes *= dim.value
	    # print(variable_parametes)
	    total_parameters += variable_parametes
	print("Total trainable model parameters:", total_parameters)
	return total_parameters


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W, stride=1, padd='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padd)


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_3x3(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_16x16(x, padd='VALID'):
	return tf.nn.avg_pool(x, ksize=[1, 16, 16, 1], strides=[1, 2, 2, 1], padding=padd)


def cnn_model_1(input_images):

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	conv1_shape = [3, 3, 3, 32]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(conv2d(input_images, W_conv1) + b_conv1)

	h_pool1 = max_pool_2x2(h_conv1)

	conv2_shape = [3, 3, 32, 32]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	h_pool2 = max_pool_2x2(h_conv2)

	conv3_shape = [3, 3, 32, 32]
	W_conv3 = weight_variable(conv3_shape)
	b_conv3 = bias_variable([conv3_shape[3]])
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

	h_pool3 = max_pool_2x2(h_conv3)

	conv4_shape = [3, 3, 32, 32]
	W_conv4 = weight_variable(conv4_shape)
	b_conv4 = bias_variable([conv4_shape[3]])
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

	h_pool4 = max_pool_2x2(h_conv4)

	conv5_shape = [3, 3, 32, 32]
	W_conv5 = weight_variable(conv5_shape)
	b_conv5 = bias_variable([conv5_shape[3]])
	h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

	h_pool5 = max_pool_2x2(h_conv5)

	fc1_shape = [8 * 8 * 32, 5]
	fc1_flat_in = tf.reshape(h_pool5, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)

	# fc1_shape = [16 * 16 * 32, 2048]
	# h_pool2_flat = tf.reshape(h_pool4, [-1, fc1_shape[0]]) # reshape to a vector
	# W_fc1 = weight_variable(fc1_shape)
	# b_fc1 = bias_variable([fc1_shape[1]])
	# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# fc2_shape = [2048, output_size]
	# W_fc2 = weight_variable(fc2_shape)
	# b_fc2 = bias_variable([fc2_shape[1]])
	# h_fc2 = (tf.matmul(h_fc1, W_fc2) + b_fc2)

	softmax = tf.nn.softmax(h_fc1)

	output = h_fc1

	return output, softmax


def cnn_model_2(input_images):

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	conv1_shape = [3, 3, 3, 32]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(conv2d(input_images, W_conv1) + b_conv1)

	h_pool1 = max_pool_2x2(h_conv1)

	conv2_shape = [3, 3, 32, 48]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	h_pool2 = max_pool_2x2(h_conv2)

	conv3_shape = [3, 3, 48, 64]
	W_conv3 = weight_variable(conv3_shape)
	b_conv3 = bias_variable([conv3_shape[3]])
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

	h_pool3 = max_pool_2x2(h_conv3)

	conv4_shape = [3, 3, 64, 64]
	W_conv4 = weight_variable(conv4_shape)
	b_conv4 = bias_variable([conv4_shape[3]])
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

	h_pool4 = max_pool_2x2(h_conv4)

	conv5_shape = [3, 3, 64, 32]
	W_conv5 = weight_variable(conv5_shape)
	b_conv5 = bias_variable([conv5_shape[3]])
	h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

	h_pool5 = max_pool_2x2(h_conv5)

	fc1_shape = [8 * 8 * 32, 1024]
	fc1_flat_in = tf.reshape(h_pool5, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = tf.nn.relu(tf.matmul(fc1_flat_in, W_fc1) + b_fc1)

	fc2_shape = [1024, 5]
	W_fc2 = weight_variable(fc2_shape)
	b_fc2 = bias_variable([fc2_shape[1]])
	h_fc2 = (tf.matmul(h_fc1, W_fc2) + b_fc2)

	output = h_fc2
	softmax = tf.nn.softmax(output)

	return output, softmax


def cnn_model_3(input_images):  # squeezenet small

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	print("===============================================================")
	print("Model Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	h_pool2 = max_pool_3x3(h_f1)  # -> 32x32
	print(h_pool2)

	# fire 2
	f2_sq_shape = [1, 1, 128, 32]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_pool2, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 32, 128]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 32, 128]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_pool3 = max_pool_3x3(h_f2)  # -> 16x16
	print(h_pool3)

	# fire 3
	f3_sq_shape = [1, 1, 256, 64]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_pool3, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 64, 256]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 64, 256]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	fc1_shape = [16 * 16 * 512, 5]
	fc1_flat_in = tf.reshape(h_f3, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	print("===============================================================")

	return output, softmax


def cnn_model_4(input_images):  # squeezenet

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	print("===============================================================")
	print("Model Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 32]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 32, 128]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 32, 128]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_pool2 = max_pool_3x3(h_f2)  # -> 32x32
	print(h_pool2)

	# fire 3
	f3_sq_shape = [1, 1, 256, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_pool2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	# fire 4
	f4_sq_shape = [1, 1, 256, 64]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_f3, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 64, 256]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 64, 256]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_pool3 = max_pool_3x3(h_f4)  # -> 16x16
	print(h_pool3)

	# fire 5
	f5_sq_shape = [1, 1, 512, 64]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_pool3, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 64, 256]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 64, 256]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	h_pool4 = max_pool_3x3(h_f5)  # -> 8x8
	print(h_pool4)

	fc1_shape = [8 * 8 * 512, 5]
	fc1_flat_in = tf.reshape(h_pool4, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	print("===============================================================")

	return output, softmax


def cnn_model_5(input_images):  # squeezenet almost exactly

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	print("===============================================================")
	print("Model 5 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	conv2_shape = [1, 1, 512, 5]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_f8, W_conv2) + b_conv2)
	print(h_conv2)

	h_pool4 = avg_pool_16x16(h_conv2)
	pool4_flat_out = tf.reshape(h_pool4, [-1, 5])

	output = pool4_flat_out
	softmax = tf.nn.softmax(output)
	print(softmax)

	print("===============================================================")

	return output, softmax


def cnn_model_6(input_images):  # squeezenet very similar

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 6 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_pool4 = max_pool_3x3(h_f8)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 512, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	print("===============================================================")

	return output, softmax


def cnn_model_7(input_images, drop_out_keep_prob):  # squeezenet very similar but smaller

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 7 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	h_pool3 = max_pool_3x3(h_f5)  # -> 16x16
	print(h_pool3)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_pool3, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_pool4 = max_pool_3x3(h_f6)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 384, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	h_conv2_drop = tf.nn.dropout(h_conv2, drop_out_keep_prob)
	print(h_conv2_drop)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2_drop, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_8(input_images, drop_out_keep_prob):  # squeezenet very similar but smaller

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 8 Architecture: ")
	print(input_images)

	conv1_shape = [5, 5, 3, 48]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 48, 8]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 8, 32]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 8, 32]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 64, 12]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 12, 48]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 12, 48]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	# fire 3
	f3_sq_shape = [1, 1, 96, 24]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 24, 96]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 24, 96]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 192, 24]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 24, 96]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 24, 96]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	# fire 5
	f5_sq_shape = [1, 1, 192, 32]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 32, 128]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 32, 128]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	h_pool3 = max_pool_3x3(h_f5)  # -> 16x16
	print(h_pool3)

	# fire 6
	f6_sq_shape = [1, 1, 256, 32]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_pool3, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 32, 128]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 32, 128]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f6_drop = tf.nn.dropout(h_f6, drop_out_keep_prob)
	print(h_f6_drop)

	h_pool4 = max_pool_3x3(h_f6_drop)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 256, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f1_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f3_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f5_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool4 = max_pool_3x3(h_f8_drop)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 512, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9_384(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 384
	image_height = 384
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 - 384 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=3) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat([h_f1_e1, h_f1_e3], 3)
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat([h_f2_e1, h_f2_e3], 3)
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f1_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat([h_f3_e1, h_f3_e3], 3)
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat([h_f4_e1, h_f4_e3], 3)
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f3_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat([h_f5_e1, h_f5_e3], 3)
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat([h_f6_e1, h_f6_e3], 3)
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f5_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat([h_f7_e1, h_f7_e3], 3)
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat([h_f8_e1, h_f8_e3], 3)
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool4 = max_pool_3x3(h_f8_drop)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 512, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9_512(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 512
	image_height = 512
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 Architecture - 512 - stride 4 start: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=4) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f1_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f3_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f5_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool4 = max_pool_3x3(h_f8_drop)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 512, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9_512_2(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 512
	image_height = 512
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 Architecture - 512 - 2 conv start: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 32]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 256x256
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 128x128
	print(h_pool1)

	conv2_shape = [3, 3, 32, 96]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=2) + b_conv2)  # -> 64x64
	print(h_conv2)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_conv2, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f1_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f3_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f5_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool4 = max_pool_3x3(h_f8_drop)  # -> 8x8
	print(h_pool4)

	conv3_shape = [1, 1, 512, 64]
	W_conv3 = weight_variable(conv3_shape)
	b_conv3 = bias_variable([conv3_shape[3]])
	h_conv3 = tf.nn.relu(conv2d(h_pool4, W_conv3) + b_conv3)
	print(h_conv3)

	fc1_shape = [8 * 8 * 64, 5]
	fc1_flat_in = tf.reshape(h_conv3, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9_768(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 768
	image_height = 768
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 - 768 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 384x384
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 192x192
	print(h_pool1)

	# fire 1
	f1_in = h_pool1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(f1_in, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_in = h_f1
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(f2_in, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_in = h_f1_f2
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(f3_in, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 96x96
	print(h_pool2)

	# fire 4
	f4_in = h_pool2
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(f4_in, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_in = h_f3_f4
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(f5_in, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_in = h_f5
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(f6_in, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_in = h_f5_f6
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(f7_in, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 48x48
	print(h_pool3)

	# fire 8
	f8_in = h_pool3
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(f8_in, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_pool4 = max_pool_3x3(h_f7_f8)  # -> 24x24
	print(h_pool4)

	# fire 9
	f9_in = h_pool4
	f9_sq_shape = [1, 1, 512, 64]
	W_f9_sq = weight_variable(f9_sq_shape)
	b_f9_sq = bias_variable([f9_sq_shape[3]])
	h_f9_sq = tf.nn.relu(conv2d(f9_in, W_f9_sq) + b_f9_sq)
	f9_e1_shape = [1, 1, 64, 256]
	W_f9_e1 = weight_variable(f9_e1_shape)
	b_f9_e1 = bias_variable([f9_e1_shape[3]])
	h_f9_e1 = tf.nn.relu(conv2d(h_f9_sq, W_f9_e1) + b_f9_e1)
	f9_e3_shape = [3, 3, 64, 256]
	W_f9_e3 = weight_variable(f9_e3_shape)
	b_f9_e3 = bias_variable([f9_e3_shape[3]])
	h_f9_e3 = tf.nn.relu(conv2d(h_f9_sq, W_f9_e3) + b_f9_e3)
	h_f9 = tf.concat(3, [h_f9_e1, h_f9_e3])
	print(h_f9)

	h_pool5 = max_pool_3x3(h_f9)  # -> 12x12
	print(h_pool5)

	# fire 10
	f10_in = h_pool5
	f10_sq_shape = [1, 1, 512, 64]
	W_f10_sq = weight_variable(f10_sq_shape)
	b_f10_sq = bias_variable([f10_sq_shape[3]])
	h_f10_sq = tf.nn.relu(conv2d(f10_in, W_f10_sq) + b_f10_sq)
	f10_e1_shape = [1, 1, 64, 256]
	W_f10_e1 = weight_variable(f10_e1_shape)
	b_f10_e1 = bias_variable([f10_e1_shape[3]])
	h_f10_e1 = tf.nn.relu(conv2d(h_f10_sq, W_f10_e1) + b_f10_e1)
	f10_e3_shape = [3, 3, 64, 256]
	W_f10_e3 = weight_variable(f10_e3_shape)
	b_f10_e3 = bias_variable([f10_e3_shape[3]])
	h_f10_e3 = tf.nn.relu(conv2d(h_f10_sq, W_f10_e3) + b_f10_e3)
	h_f10 = tf.concat(3, [h_f10_e1, h_f10_e3])
	print(h_f10)

	h_f9_f10 = h_pool5 + h_f10  # bypass
	print(h_f9_f10)

	h_f10_drop = tf.nn.dropout(h_f9_f10, drop_out_keep_prob)
	print(h_f10_drop)

	h_pool6 = max_pool_3x3(h_f10_drop)  # -> 6x6
	print(h_pool6)

	conv2_shape = [1, 1, 512, 96]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool6, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [6 * 6 * 96, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_9_768_2(input_images, drop_out_keep_prob):  # squeezenet simple bypass

	image_width = 768
	image_height = 768
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 9 - 768 - Smaller Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 384x384
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 192x192
	print(h_pool1)

	# fire 1
	f1_in = h_pool1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(f1_in, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_in = h_f1
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(f2_in, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_in = h_f1_f2
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(f3_in, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 96x96
	print(h_pool2)

	# fire 4
	f4_in = h_pool2
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(f4_in, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_in = h_f3_f4
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(f5_in, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	h_pool3 = max_pool_3x3(h_f5_f6)  # -> 48x48
	print(h_pool3)

	# fire 6
	f6_in = h_pool3
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(f6_in, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_pool3 + h_f6  # bypass
	print(h_f5_f6)

	h_pool4 = max_pool_3x3(h_f5_f6)  # -> 24x24
	print(h_pool4)

	# fire 7
	f7_in = h_pool4
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(f7_in, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool5 = max_pool_3x3(h_f7)  # -> 12x12
	print(h_pool5)

	# fire 8
	f8_in = h_pool5
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(f8_in, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool5 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool6 = max_pool_3x3(h_f8_drop)  # -> 6x6
	print(h_pool6)

	conv2_shape = [1, 1, 512, 96]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool6, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [6 * 6 * 96, 5]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	output = h_fc1
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


def cnn_model_10(input_images, drop_out_keep_prob):

	image_width = 256
	image_height = 256
	image_depth = 3
	output_size = 5

	# save architecure to file
	print("===============================================================")
	print("Model 10 Architecture: ")
	print(input_images)

	conv1_shape = [7, 7, 3, 96]
	W_conv1 = weight_variable(conv1_shape)
	b_conv1 = bias_variable([conv1_shape[3]])
	h_conv1 = tf.nn.relu(
		conv2d(input_images, W_conv1, stride=2) + b_conv1)  # -> 128x128
	print(h_conv1)

	h_pool1 = max_pool_3x3(h_conv1)  # -> 64x64
	print(h_pool1)

	# fire 1
	f1_sq_shape = [1, 1, 96, 16]
	W_f1_sq = weight_variable(f1_sq_shape)
	b_f1_sq = bias_variable([f1_sq_shape[3]])
	h_f1_sq = tf.nn.relu(conv2d(h_pool1, W_f1_sq) + b_f1_sq)
	f1_e1_shape = [1, 1, 16, 64]
	W_f1_e1 = weight_variable(f1_e1_shape)
	b_f1_e1 = bias_variable([f1_e1_shape[3]])
	h_f1_e1 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e1) + b_f1_e1)
	f1_e3_shape = [3, 3, 16, 64]
	W_f1_e3 = weight_variable(f1_e3_shape)
	b_f1_e3 = bias_variable([f1_e3_shape[3]])
	h_f1_e3 = tf.nn.relu(conv2d(h_f1_sq, W_f1_e3) + b_f1_e3)
	h_f1 = tf.concat(3, [h_f1_e1, h_f1_e3])
	print(h_f1)

	# fire 2
	f2_sq_shape = [1, 1, 128, 16]
	W_f2_sq = weight_variable(f2_sq_shape)
	b_f2_sq = bias_variable([f2_sq_shape[3]])
	h_f2_sq = tf.nn.relu(conv2d(h_f1, W_f2_sq) + b_f2_sq)
	f2_e1_shape = [1, 1, 16, 64]
	W_f2_e1 = weight_variable(f2_e1_shape)
	b_f2_e1 = bias_variable([f2_e1_shape[3]])
	h_f2_e1 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e1) + b_f2_e1)
	f2_e3_shape = [3, 3, 16, 64]
	W_f2_e3 = weight_variable(f2_e3_shape)
	b_f2_e3 = bias_variable([f2_e3_shape[3]])
	h_f2_e3 = tf.nn.relu(conv2d(h_f2_sq, W_f2_e3) + b_f2_e3)
	h_f2 = tf.concat(3, [h_f2_e1, h_f2_e3])
	print(h_f2)

	h_f1_f2 = h_f1 + h_f2  # bypass
	print(h_f1_f2)

	# fire 3
	f3_sq_shape = [1, 1, 128, 32]
	W_f3_sq = weight_variable(f3_sq_shape)
	b_f3_sq = bias_variable([f3_sq_shape[3]])
	h_f3_sq = tf.nn.relu(conv2d(h_f1_f2, W_f3_sq) + b_f3_sq)
	f3_e1_shape = [1, 1, 32, 128]
	W_f3_e1 = weight_variable(f3_e1_shape)
	b_f3_e1 = bias_variable([f3_e1_shape[3]])
	h_f3_e1 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e1) + b_f3_e1)
	f3_e3_shape = [3, 3, 32, 128]
	W_f3_e3 = weight_variable(f3_e3_shape)
	b_f3_e3 = bias_variable([f3_e3_shape[3]])
	h_f3_e3 = tf.nn.relu(conv2d(h_f3_sq, W_f3_e3) + b_f3_e3)
	h_f3 = tf.concat(3, [h_f3_e1, h_f3_e3])
	print(h_f3)

	h_pool2 = max_pool_3x3(h_f3)  # -> 32x32
	print(h_pool2)

	# fire 4
	f4_sq_shape = [1, 1, 256, 32]
	W_f4_sq = weight_variable(f4_sq_shape)
	b_f4_sq = bias_variable([f4_sq_shape[3]])
	h_f4_sq = tf.nn.relu(conv2d(h_pool2, W_f4_sq) + b_f4_sq)
	f4_e1_shape = [1, 1, 32, 128]
	W_f4_e1 = weight_variable(f4_e1_shape)
	b_f4_e1 = bias_variable([f4_e1_shape[3]])
	h_f4_e1 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e1) + b_f4_e1)
	f4_e3_shape = [3, 3, 32, 128]
	W_f4_e3 = weight_variable(f4_e3_shape)
	b_f4_e3 = bias_variable([f4_e3_shape[3]])
	h_f4_e3 = tf.nn.relu(conv2d(h_f4_sq, W_f4_e3) + b_f4_e3)
	h_f4 = tf.concat(3, [h_f4_e1, h_f4_e3])
	print(h_f4)

	h_f3_f4 = h_pool2 + h_f4  # bypass
	print(h_f3_f4)

	# fire 5
	f5_sq_shape = [1, 1, 256, 48]
	W_f5_sq = weight_variable(f5_sq_shape)
	b_f5_sq = bias_variable([f5_sq_shape[3]])
	h_f5_sq = tf.nn.relu(conv2d(h_f3_f4, W_f5_sq) + b_f5_sq)
	f5_e1_shape = [1, 1, 48, 192]
	W_f5_e1 = weight_variable(f5_e1_shape)
	b_f5_e1 = bias_variable([f5_e1_shape[3]])
	h_f5_e1 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e1) + b_f5_e1)
	f5_e3_shape = [3, 3, 48, 192]
	W_f5_e3 = weight_variable(f5_e3_shape)
	b_f5_e3 = bias_variable([f5_e3_shape[3]])
	h_f5_e3 = tf.nn.relu(conv2d(h_f5_sq, W_f5_e3) + b_f5_e3)
	h_f5 = tf.concat(3, [h_f5_e1, h_f5_e3])
	print(h_f5)

	# fire 6
	f6_sq_shape = [1, 1, 384, 48]
	W_f6_sq = weight_variable(f6_sq_shape)
	b_f6_sq = bias_variable([f6_sq_shape[3]])
	h_f6_sq = tf.nn.relu(conv2d(h_f5, W_f6_sq) + b_f6_sq)
	f6_e1_shape = [1, 1, 48, 192]
	W_f6_e1 = weight_variable(f6_e1_shape)
	b_f6_e1 = bias_variable([f6_e1_shape[3]])
	h_f6_e1 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e1) + b_f6_e1)
	f6_e3_shape = [3, 3, 48, 192]
	W_f6_e3 = weight_variable(f6_e3_shape)
	b_f6_e3 = bias_variable([f6_e3_shape[3]])
	h_f6_e3 = tf.nn.relu(conv2d(h_f6_sq, W_f6_e3) + b_f6_e3)
	h_f6 = tf.concat(3, [h_f6_e1, h_f6_e3])
	print(h_f6)

	h_f5_f6 = h_f5 + h_f6  # bypass
	print(h_f5_f6)

	# fire 7
	f7_sq_shape = [1, 1, 384, 64]
	W_f7_sq = weight_variable(f7_sq_shape)
	b_f7_sq = bias_variable([f7_sq_shape[3]])
	h_f7_sq = tf.nn.relu(conv2d(h_f6, W_f7_sq) + b_f7_sq)
	f7_e1_shape = [1, 1, 64, 256]
	W_f7_e1 = weight_variable(f7_e1_shape)
	b_f7_e1 = bias_variable([f7_e1_shape[3]])
	h_f7_e1 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e1) + b_f7_e1)
	f7_e3_shape = [3, 3, 64, 256]
	W_f7_e3 = weight_variable(f7_e3_shape)
	b_f7_e3 = bias_variable([f7_e3_shape[3]])
	h_f7_e3 = tf.nn.relu(conv2d(h_f7_sq, W_f7_e3) + b_f7_e3)
	h_f7 = tf.concat(3, [h_f7_e1, h_f7_e3])
	print(h_f7)

	h_pool3 = max_pool_3x3(h_f7)  # -> 16x16
	print(h_pool3)

	# fire 8
	f8_sq_shape = [1, 1, 512, 64]
	W_f8_sq = weight_variable(f8_sq_shape)
	b_f8_sq = bias_variable([f8_sq_shape[3]])
	h_f8_sq = tf.nn.relu(conv2d(h_pool3, W_f8_sq) + b_f8_sq)
	f8_e1_shape = [1, 1, 64, 256]
	W_f8_e1 = weight_variable(f8_e1_shape)
	b_f8_e1 = bias_variable([f8_e1_shape[3]])
	h_f8_e1 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e1) + b_f8_e1)
	f8_e3_shape = [3, 3, 64, 256]
	W_f8_e3 = weight_variable(f8_e3_shape)
	b_f8_e3 = bias_variable([f8_e3_shape[3]])
	h_f8_e3 = tf.nn.relu(conv2d(h_f8_sq, W_f8_e3) + b_f8_e3)
	h_f8 = tf.concat(3, [h_f8_e1, h_f8_e3])
	print(h_f8)

	h_f7_f8 = h_pool3 + h_f8  # bypass
	print(h_f7_f8)

	h_f8_drop = tf.nn.dropout(h_f7_f8, drop_out_keep_prob)
	print(h_f8_drop)

	h_pool4 = max_pool_3x3(h_f8_drop)  # -> 8x8
	print(h_pool4)

	conv2_shape = [1, 1, 512, 64]
	W_conv2 = weight_variable(conv2_shape)
	b_conv2 = bias_variable([conv2_shape[3]])
	h_conv2 = tf.nn.relu(conv2d(h_pool4, W_conv2) + b_conv2)
	print(h_conv2)

	fc1_shape = [8 * 8 * 64, 256]
	fc1_flat_in = tf.reshape(h_conv2, [-1, fc1_shape[0]])
	W_fc1 = weight_variable(fc1_shape)
	b_fc1 = bias_variable([fc1_shape[1]])
	h_fc1 = (tf.matmul(fc1_flat_in, W_fc1) + b_fc1)
	print(h_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1, drop_out_keep_prob)
	print(h_fc1_drop)

	fc2_shape = [256, 5]
	W_fc2 = weight_variable(fc2_shape)
	b_fc2 = bias_variable([fc2_shape[1]])
	h_fc2 = (tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	print(h_fc2)

	output = h_fc2
	softmax = tf.nn.softmax(output)
	print(softmax)

	show_num_parameters()

	print("===============================================================")

	return output, softmax


# buffer space
