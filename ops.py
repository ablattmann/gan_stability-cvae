import tensorflow as tf


def upsample(x, name=None):
    _, nh, nw, nx = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, [nh * 2, nw * 2], name=name)
    return x


def resnet_block(input_, fin, fhidden, fout, name='rnb', actfcn=tf.nn.leaky_relu):
    '''

    :param fin:
    :param fhidden:
    :param fout:
    :return:
    '''
    with tf.variable_scope(name):
        assert input_.get_shape()[-1] == fin
        x0 = input_
        x = conv2d(actfcn(input_), fhidden, 3, 3, 1, 1, 'conv2d_1')
        x = conv2d(actfcn(x), fout, 3, 3, 1, 1, 'conv2d_2')
        x0 = conv2d(x0, fout, 1, 1, 1, 1, 'conv2d_sc')

        return x0 + 0.1 * x


def resnet_block_cin(input_, labels, fin, fhidden, fout, n_classes, name='rnb_cin', is_training=True,
                     actfcn=tf.nn.leaky_relu):
    '''
    This function implements a resnet block with conditional instance normalization
    :param input:
    :param labels:
    :param fin:
    :param fhidden:
    :param fout:
    :param n_classes:
    :param name:
    :param is_training:
    :param actfcn:
    :return:
    '''
    with tf.variable_scope(name):
        assert input_.get_shape()[-1] == fin
        cin_1 = ConditionalInstanceNorm(n_classes, name='cin_1')
        cin_2 = ConditionalInstanceNorm(n_classes, name='cin_2')
        x0 = input_
        x = actfcn(cin_1(input_, labels, is_training))
        x = conv2d(x, fhidden, 3, 3, 1, 1, 'conv2d_1')
        x = actfcn(cin_2(x, labels, is_training))
        x = conv2d(x, fout, 3, 3, 1, 1, 'conv2d_2')
        x0 = conv2d(x0, fout, 1, 1, 1, 1, 'conv2d_sc')

        return x0 + 0.1 * x


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, name='conv2d', preserve_size=True):
    """Creates convolutional layers which use xavier initializer.
    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_dim: Number of features in the output layer.
      k_h: The height of the convolutional kernel.
      k_w: The width of the convolutional kernel.
      d_h: The height stride of the convolutional kernel.
      d_w: The width stride of the convolutional kernel.
      name: The name of the variable scope.
      preserve_size: whether to preserve spatial size of filter kernel
    Returns:
      conv: The normalized tensor.
    """
    with tf.variable_scope(name):
        if (preserve_size):
            pad = 'SAME'
        else:
            pad = 'VALID'

        w = tf.get_variable(
            'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, biases)
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name='deconv2d', init_bias=0.):
    """Creates deconvolutional layers.
    Args:
      input_: 4D input tensor (batch size, height, width, channel).
      output_shape: Number of features in the output layer.
      k_h: The height of the convolutional kernel.
      k_w: The width of the convolutional kernel.
      d_h: The height stride of the convolutional kernel.
      d_w: The width stride of the convolutional kernel.
      stddev: The standard deviation for weights initializer.
      name: The name of the variable scope.
      init_bias: The initial bias for the layer.
    Returns:
      conv: The normalized tensor.
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
                                 initializer=tf.constant_initializer(init_bias))
        deconv = tf.nn.bias_add(deconv, biases)
        deconv.shape.assert_is_compatible_with(output_shape)

        return deconv


def linear(x, output_size, scope=None, bias_zero=True):
    """Creates a linear layer.
    Args:
      x: 2D input tensor (batch size, features)
      output_size: Number of features in the output layer
      scope: Optional, variable scope to put the layer's parameters into
    Returns:
      The normalized tensor
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable(
            'Matrix', [shape[1], output_size], tf.float32,
            tf.contrib.layers.xavier_initializer())
        if bias_zero:
            bias_start = 0.0
        else:
            bias_start = 1.0 / shape[1]
        bias = tf.get_variable(
            'bias', [output_size], initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(x, matrix) + bias
        return out


def lrelu(x, leak=0.2, name='lrelu'):
    """The leaky RELU operation."""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def embedding(x, input_size, embed_size, name='embedding'):
    with tf.variable_scope(name):
        # Create lookup table
        embedding_map = tf.get_variable(shape=[input_size, embed_size], name='embedding_map',
                                        initializer=tf.contrib.layers.xavier_initializer())
        # Create Embedding
        lt = tf.nn.embedding_lookup(embedding_map, x)
        return lt


class ConditionalInstanceNorm(object):

    def __init__(self, n_classes, name='conditional_instance_norm', moving_average=0.99, scale=True, shift=True,
                 labels_one_hot=False):
        '''

        :param n_classes:
        :param name:
        :param moving_average:
        :param scale:
        :param shift:
        '''
        self.name = name
        with tf.variable_scope(self.name):
            self.n_classes = n_classes
            self.moving_average = moving_average
            self.scale = scale
            self.shift = shift
            self.labels_one_hot = labels_one_hot

    def __call__(self, input, labels, is_training):
        '''

        :param input:
        :param labels:
        :param is_training:
        :return:
        '''
        # define the shape of the learned weighting matrices betaand gamma
        trainable_shape = tf.TensorShape([self.n_classes, input.get_shape()[-1]])
        mvg_average_shape = tf.TensorShape([1, 1, 1, input.get_shape()[-1]])

        with tf.variable_scope(self.name):
            # define the translation part of the affine transformation
            self.beta = tf.get_variable(name='beta', shape=trainable_shape, dtype=tf.float32,
                                        initializer=tf.zeros_initializer(), trainable=True)

            # define the rotational part of the affine transformation
            self.gamma = tf.get_variable(name='gamma', shape=trainable_shape, dtype=tf.float32,
                                         initializer=tf.ones_initializer(), trainable=True)

            # define moving_averages
            self.mvg_mean = tf.get_variable(name='moving_mean', shape=mvg_average_shape, dtype=tf.float32,
                                            initializer=tf.zeros_initializer(), trainable=False)
            self.mvg_variance = tf.get_variable(name='moving_sigma', shape=mvg_average_shape, dtype=tf.float32,
                                                initializer=tf.ones_initializer(), trainable=False)

            # operations needed for the training or update
            # if(self.labels_one_hot):
            #     beta = tf.gather_nd()
            # else:
            beta = tf.gather(self.beta, labels)
            beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)

            gamma = tf.gather(self.gamma, labels)
            gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)

            moving_avg = self.moving_average

            # training and inference are handled differently
            if (is_training):
                # calculate moments
                mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=True)

                # update moving mean and variance if in training mode; IMPORTANT: ADD THE UPDATE OPERATION TO tf.GraphKeys.UPDATE_OPS
                # it is also important to define a context manager before calling the train operation
                update_mvg_mean = tf.assign(self.mvg_mean,
                                            self.mvg_mean * moving_avg + mean * (1 - moving_avg))
                update_mvg_variance = tf.assign(self.mvg_variance,
                                                self.mvg_variance * moving_avg + variance * (1 - moving_avg))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mvg_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mvg_variance)

                # apply batch_norm
                output = tf.nn.batch_normalization(input, mean, variance, beta, gamma, variance_epsilon=1E-5)
            else:
                output = tf.nn.batch_normalization(input, self.mvg_mean, self.mvg_variance, beta, gamma,
                                                   variance_epsilon=1E-5)

            return output


class RecyclableLinearLayer(object):

    def __init__(self, n_out, name, bias_zero_init):
        '''
        This implements a common linear layer that is recyclable
        :param n_out: the output dimension
        :param name: name of the layer
        :param bias_zero_init: whether to initialize the bias parameters with zero or not
        '''
        with tf.variable_scope(name):
            self.name = name
            self.n_out = n_out
            self.bias_zero_init = bias_zero_init

    def __call__(self, input):
        '''
        Call function is used, for creation of operations
        :param input: the input tensor
        :return:
        '''
        shape = input.get_shape().as_list()

        with tf.variable_scope(self.name):
            output_size = self.n_out
            bias_zero = self.bias_zero_init

            # common linear layer implementation
            matrix = tf.get_variable(
                'Matrix', [shape[1], output_size], tf.float32,
                tf.contrib.layers.xavier_initializer())
            if bias_zero:
                bias_start = 0.0
            else:
                bias_start = 1.0 / shape[1]
            bias = tf.get_variable(
                'bias', [output_size], initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(input, matrix) + bias
            return out
