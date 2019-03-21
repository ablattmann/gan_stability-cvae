import tensorflow as tf
import ops


def condition_embedded_generator(z_g, target_labels, z_g_dim, n_classes, nf_init, n_stages,
                                 name='generator_embedded'):
    '''

    :param z_g: batch of random vector of size batch_size x z_g_dim
    :param target_labels: batch of target_labels, encoded as integers pointing out the label index
    :param z_g_dim: dimension of a single the noise vector z_g
    :param n_classes: number of classes, dimension of a single label vector
    :param nf_init: number of initial filter kernels in the generator
    :param n_stages: number of consequtively stacked resnet-blocks
    :param name: name of the generator scope
    :return: a batch of generated data points of size batch_size x width x height x 3
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # create label embedding
        label_embedding = ops.embedding(target_labels, n_classes, z_g_dim, 'g_label_embedding')

        label_embedding = tf.nn.l2_normalize(label_embedding, axis=1)

        assert label_embedding.get_shape() == z_g.get_shape()

        # concatenate embedding with input
        x = tf.concat([z_g, label_embedding], axis=1)
        x = ops.linear(x, 4 * 4 * nf_init * (2 ** (n_stages - 1)), 'g_linear_in', bias_zero=False)
        x = tf.reshape(x, [-1, 4, 4, nf_init * (2 ** (n_stages - 1))])  # 4x4

        # First resnet block preserves depth
        x = ops.resnet_block(x, nf_init * (2 ** (n_stages - 1)), nf_init * (2 ** (n_stages - 1)),
                             nf_init * (2 ** (n_stages - 1)), 'g_rnb_1')
        x = ops.upsample(x, name='g_upsample_1')  # 8x8 spatial dim

        for i in range(n_stages - 1):
            # Resnet Blocks and upsampling
            x = ops.resnet_block(x, nf_init * (2 ** (n_stages - i - 1)), nf_init * (2 ** (n_stages - i - 2)),
                                 nf_init * (2 ** (n_stages - i - 2)), 'g_rnb_{}'.format(2 + i))
            x = ops.upsample(x, name='g_upsample_{}'.format(2 + i))

        # Last convolution to output image
        x = tf.nn.leaky_relu(x)
        x = ops.conv2d(x, 3, 3, 3, 1, 1, 'g_conv_image')

        x = tf.nn.tanh(x, name='g_final_act')
        return x


def conditioned_generator(z_g, target_labels, nf_init, n_stages,
                          name='generator'):
    '''

    :param z_g: batch of random vector of size batch_size x z_g_dim
    :param target_labels: batch of target_labels, encoded as one hot vectors
    :param nf_init: number of initial filter kernels in the generator
    :param n_stages: number of consequtively stacked resnet-blocks
    :param name: name of the generator scope
    :return: a batch of generated data points of size batch_size x width x height x 3
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # create label embedding

        # concatenate embedding with input
        x = tf.concat([z_g, target_labels], axis=1)
        x = ops.linear(x, 4 * 4 * nf_init * (2 ** (n_stages - 1)), 'g_linear_in', bias_zero=False)
        x = tf.reshape(x, [-1, 4, 4, nf_init * (2 ** (n_stages - 1))])  # 4x4

        # First resnet block preserves depth
        x = ops.resnet_block(x, nf_init * (2 ** (n_stages - 1)), nf_init * (2 ** (n_stages - 1)),
                             nf_init * (2 ** (n_stages - 1)), 'g_rnb_1')

        x = ops.upsample(x, name='g_upsample_1')  # 8x8 spatial dim

        for i in range(n_stages - 1):
            # Resnet Blocks and upsampling
            x = ops.resnet_block(x, nf_init * (2 ** (n_stages - i - 1)), nf_init * (2 ** (n_stages - i - 2)),
                                 nf_init * (2 ** (n_stages - i - 2)), 'g_rnb_{}'.format(2 + i))
            x = ops.upsample(x, name='g_upsample_{}'.format(2 + i))

        # Last convolution to output image
        x = tf.nn.leaky_relu(x)
        x = ops.conv2d(x, 3, 3, 3, 1, 1, 'g_conv_image')

        x = tf.nn.tanh(x, name='g_final_act')
        return x


def conditional_instance_normalized_generator(z_g, target_labels, n_classes, nf_init, n_stages,
                                              name='generator_cin', is_training=True):
    '''
    Generator based on Resnet-architecture, with conditional instance norm in resnet blocks: Learns a mapping from an input
    tensor thats sampled from a normal distribution to an image that shows the label depicted by the target labels vector
    :param z_g: random sampled input noise tensor of shape (batch_size, noise_dim)
    :param target_labels: 2D integer tensor of shape (batch_size,)
    :param n_classes: scalar depticting the actual number of classes
    :param nf_init: number of filters in last generator layer before final convolution to 3 image channels is performed
    :param n_stages: number of consequetivly stacked Resnet Blocks
    :param name: depicting the name of the scope which all generator ops are placed in
    :param is_training: flag depticing if generator is in training or inference mode (required for conditional instance norm)
    :return:
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = ops.linear(z_g, 4 * 4 * nf_init * (2 ** (n_stages - 1)), 'g_linear_in', bias_zero=False)
        x = tf.reshape(x, [-1, 4, 4, nf_init * (2 ** (n_stages - 1))])  # 4x4

        # First resnet block preserves depth
        x = ops.resnet_block_cin(x, target_labels, nf_init * (2 ** (n_stages - 1)), nf_init * (2 ** (n_stages - 1)),
                                 nf_init * (2 ** (n_stages - 1)), n_classes, 'g_rnb_1', is_training)
        x = ops.upsample(x, name='g_upsample_1')  # 8x8 spatial dim

        for i in range(n_stages - 1):
            # Resnet Blocks and upsampling
            x = ops.resnet_block_cin(x, target_labels, nf_init * (2 ** (n_stages - i - 1)),
                                     nf_init * (2 ** (n_stages - i - 2)),
                                     nf_init * (2 ** (n_stages - i - 2)), n_classes, 'g_rnb_{}'.format(2 + i),
                                     is_training)
            x = ops.upsample(x, name='g_upsample_{}'.format(2 + i))

        # Last convolution to output image
        x = tf.nn.leaky_relu(x)
        x = ops.conv2d(x, 3, 3, 3, 1, 1, 'g_conv_image')

        x = tf.nn.tanh(x, name='g_final_act')

        return x
