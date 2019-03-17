import tensorflow as tf
import ops

def conditioned_discriminator(image, target_labels, n_classes, nf_start, n_stages,
                                 name='discriminator',activate_mean=False):
    '''
    Conditioned discriminator, regularized with information bottleneck according to https://arxiv.org/pdf/1810.00821.pdf
    :param image: The input image for the model
    :param target_labels: The target labels, encoded as one-hot-vectors
    :param n_classes: Number of classes in the model
    :param nf_start: Initial number of filter kernels to be used
    :param n_stages: Number of conseqeutively stacked resnet blocks in the model
    :param name: Name of the scope.that's tp be used. Recommended to leave it to be the default value
    :mean_mode: Whether to output expectation activations as well
    :return: Batch of binar classifications whether image is true or fake
    '''
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # first filter stage from image to nf_start
        x=ops.conv2d(image,nf_start,3,3,1,1,'d_conv_img')

        x = ops.resnet_block(x,nf_start,nf_start,nf_start,name='d_rnb_1')

        # apply average pooling and then stack consecutive resnet blocks
        for i in range(n_stages-1):
            # Average pool and resnet block
            x = tf.nn.avg_pool(x,[1,3,3,1],[1,2,2,1],padding='SAME',name='d_avg_pool_{}'.format(1+i))
            x = ops.resnet_block(x,nf_start * (2**(i)),nf_start * (2**(i)),nf_start * (2**(i+1)),name='d_rnb_{}'.format(2+i))

        x = tf.nn.avg_pool(x,[1,3,3,1],[1,2,2,1],padding='SAME',name='d_avg_pool_{}'.format(n_stages))

        assert x.get_shape()[-1] == nf_start * (2**(n_stages-1))

        # activate
        x = tf.nn.leaky_relu(x)

        # 1x1-2D-convolution as mentioned in https://arxiv.org/pdf/1810.00821.pdf
        parameters = ops.conv2d(x,nf_start * (2**(n_stages-1)),4,4,1,1,name='d_latent_converter',preserve_size=False)

        # remove indices equal to one
        parameters=tf.squeeze(parameters,axis=[1,2],name='d_squeeze')

        # split in order to obtain statistics
        mus, sigmas = tf.split(parameters,2,axis=1)

        # sigmas need to be in between 0 and 1
        sigmas = tf.nn.sigmoid(sigmas)

        # in mean mode --> sample from distribution given by mus and sigmas (applied, in discriminator training-forward pass)
        out = tf.random_normal(mus.get_shape().as_list(), name='d_sample_unit_normal')
        out = tf.multiply(out, sigmas, name='d_scale_variance')
        out = tf.add(out, mus, name='d_shift_expectation')

        last_linear = ops.RecyclableLinearLayer(n_classes,name='d_linear_out',bias_zero_init=False)

        # last layer, fully connected
        out = tf.nn.leaky_relu(out)
        out = last_linear(out)
        # out = ops.linear(out,n_classes,scope='d_linear_out',bias_zero=False)
        out = tf.gather_nd(out,target_labels)

        # return output, mus and sigmas
        if activate_mean:
            # if activate mean option enabled, also output mu_activations for generator optimization
            # IMPORTANT: the same fully connected layer has to be used as in the not activated mean case
            gen_out = tf.nn.leaky_relu(mus)
            gen_out = last_linear(gen_out)
            # gen_out = ops.linear(gen_out, n_classes, scope='d_linear_out', bias_zero=False)
            gen_out = tf.gather_nd(gen_out,target_labels)

            return out, mus, sigmas, gen_out
        else:
            return out, mus, sigmas


def conditioned_encoder(image, target_labels,n_classes,nf_start,n_stages,name= 'encoder',is_training=True):
    '''

    :param image:
    :param target_labels: batch of target_labels, encoded as integers pointing out the label index
    :param n_classes:
    :param nf_start:
    :param n_stages:
    :param name:
    :param is_training: Whether in training or inference mode, but since the encoder network is usually not used
    during inference time, it is right now not intended to change that parameter
    :return:
    '''
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # start using deeper filter sizes
        x=ops.conv2d(image,nf_start,3,3,1,1,'e_conv_img')

        x = ops.resnet_block_cin(x,target_labels,nf_start,nf_start,nf_start,n_classes,name='e_rnb_1',is_training=is_training)

        # apply average pooling and then stack consecutive resnet blocks
        for i in range(n_stages-1):
            # Average pool and resnet block
            x = tf.nn.avg_pool(x,[1,3,3,1],[1,2,2,1],padding='SAME',name='e_avg_pool_{}'.format(1+i))
            x = ops.resnet_block_cin(x,target_labels,nf_start * (2**(i)),nf_start * (2**(i)),nf_start * (2**(i+1)),n_classes,name='e_rnb_{}'.format(2+i),is_training=is_training)

        x = tf.nn.avg_pool(x,[1,3,3,1],[1,2,2,1],padding='SAME',name='e_avg_pool_{}'.format(n_stages))

        assert x.get_shape()[-1] == nf_start * (2**(n_stages-1))

        # activate
        x = tf.nn.leaky_relu(x)

        # 1x1-2D-convolution as mentioned in https://arxiv.org/pdf/1810.00821.pdf
        parameters = ops.conv2d(x,nf_start * (2**(n_stages-1)),4,4,1,1,name='e_latent_converter',preserve_size=False)

        # remove indices equal to one
        parameters=tf.squeeze(parameters,axis=[1,2],name='e_squeeze')

        # split in order to obtain statistics
        mus, sigmas = tf.split(parameters,2,axis=1)

        # sigmas need to be in between 0 and 1
        sigmas = tf.nn.sigmoid(sigmas)

        out = tf.random_normal(mus.get_shape().as_list(), name='e_sample_unit_normal')
        out = tf.multiply(out, sigmas, name='e_scale_variance')
        out = tf.add(out, mus, name='e_shift_expectation')

        return out, mus, sigmas
