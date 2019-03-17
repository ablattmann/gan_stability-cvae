import os
import tensorflow as tf
from absl import flags
import utils

import generator as gen_module
import discriminator as disc_module

gfile = tf.gfile
tfgan = tf.contrib.gan

model_dir = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_string('data_dir', os.path.join(model_dir, '../../data/wiki-art'),
                    'Directory, where the data is expected to be found.')
flags.DEFINE_float('g_learning_rate', 0.0001, 'The learning rate for the generator (rmsprop).')
flags.DEFINE_float('d_learning_rate', 0.0001, 'The learning rate for the discriminator (rmsprop).')
flags.DEFINE_float('gd_gamma', 0.99, 'The discouting factor for the history/coming gradient.')
flags.DEFINE_integer('image_size', 128, 'The size of image to use '
                                        '(will be center cropped) [128]')
flags.DEFINE_integer('z_g_dim', 128, 'Dimensionality of latent code z_g for the generator.')
flags.DEFINE_integer('gf_dim', 32, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('df_dim', 32, 'Dimensionality of df. [64]')
flags.DEFINE_integer('n_stages', 5, 'Number of stacked Resnet-Blocks, both in Generator and Discriminator.')
flags.DEFINE_integer('n_classes', 14, 'Number of classes that are present in the data set.')

flags.DEFINE_float('information_upper_bound', 0.3,
                   'The value used to bound the information in the information bottleneck with.')
flags.DEFINE_float('beta_step_size', 0.00001, 'Step size for optimizing the beta regularizer parameter.')
flags.DEFINE_integer('dataset_size', 90000, 'Number of training exmaples in the dataset.')
flags.DEFINE_bool('use_gradient_penalty', False,
                  'Whether to regularize the discriminator with an additional one-sided gradient penalty.')
flags.DEFINE_float('gp_gamma', 10, 'Weight for the gradient penalty.')
flags.DEFINE_bool('use_cin', True,
                  'Whether to use conditional_instance_normalization for the generator or not.')

FLAGS = flags.FLAGS


def _gradient_penalty(logits, data, gamma_gp):
    '''
    Gradient penalty loss as introduced in https://arxiv.org/abs/1801.04406
    :param logits: discriminator logits, dependent variables for the gradient computation
    :param data: the input data, the variables with respect to which the gradients are computed
    :param gamma_gp: The gradient penalty weight
    :return: weighted gradient penalty loss
    '''
    grads_squared_norm = tf.pow(tf.gradients(tf.reduce_sum(logits, axis=0), data)[0], 2, name='grads_squared_norm')
    grads_squared_norm = tf.reduce_sum(tf.reshape(grads_squared_norm, [data.get_shape()[0], -1]), axis=1)
    return gamma_gp * tf.reduce_mean(grads_squared_norm, name='gp_loss')


def _get_disc_on_real_loss(disc_on_data_logits):
    '''
    Standard cross entropy loss for discriminator on real data points
    :param disc_on_data_logits: logits of discriminator on real data
    :return: loss of discriminator classifying real examples as a scalar value
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_on_data_logits, labels=tf.ones(
        shape=disc_on_data_logits.get_shape().as_list())), name='d_real_loss')


def _get_disc_on_gen_loss(disc_on_gen_logits):
    '''
    Standard cross entropy loss for discriminator on data points from generator
    :param disc_on_gen_logits:logits of discriminator on data from generator
    :return: loss of discriminator classifying fake examples as a scalar value
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_on_gen_logits, labels=tf.zeros(
        shape=disc_on_gen_logits.get_shape().as_list())), name='d_fake_loss')


def _bottleneck_loss(mus, sigmas, i_c, name='bnc_loss'):
    '''
    Bottleneck loss as described in https://arxiv.org/pdf/1810.00821.pdf
    :param mus: Expectations of encoder network E(z|x)
    :param sigmas: Variances of encoder network E(z|x)
    :param i_c: upper limit of encoded information that's provided for the discriminator
    :return: The bottleneck loss as a scalar value
    '''
    with tf.variable_scope(name):
        return tf.reduce_mean(tf.reduce_sum(mus ** 2 + sigmas ** 2 - tf.log(sigmas ** 2) - 1, axis=1)) - i_c


def _gen_loss(disc_on_gen_mu_activations):
    '''
    Standard cross entropy loss for generator with inverted targets, in order to fool the discriminator
    :param disc_on_gen_mu_activations: activations on the mus of the encoder network E(z|x)
    :return: The loss for the generator, as a scalar value
    '''
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_on_gen_mu_activations, labels=tf.ones(
        shape=disc_on_gen_mu_activations.get_shape().as_list())), name='g_loss')


class VDBGAN:

    def __init__(self, z_g_dist, global_step, config, device=None):
        # Global optimization step
        self.global_step = global_step

        # parameters defined in train script
        self.config = config

        # device to place the operations on
        self.device = device

        # image properties
        self.image_size = FLAGS.image_size
        self.image_shape = [FLAGS.image_size, FLAGS.image_size, 3]

        # Filter depths
        self.gf_dim = FLAGS.gf_dim
        self.df_dim = FLAGS.df_dim

        # VDB properties
        self.information_constraint = FLAGS.information_upper_bound
        self.beta_step_size = FLAGS.beta_step_size

        # input data for the generator
        self.z_g_dist = z_g_dist

        # build model
        self._build()

    def _build(self):
        '''
        Build up the vdbgan-model, defines the graph structure, that's to be executed later on
        '''

        config = self.config

        # Define optimtizers
        self.g_learning_rate = FLAGS.g_learning_rate
        self.d_learning_rate = FLAGS.d_learning_rate

        # Place on gpu if available
        with tf.device(self.device):
            # define beta
            self.beta = tf.get_variable('beta', trainable=False, initializer=tf.zeros_initializer(), shape=())

            self.g_opt = tf.train.RMSPropOptimizer(self.g_learning_rate, FLAGS.gd_gamma, name='g_optimizer',epsilon=1e-8)
            self.d_opt = tf.train.RMSPropOptimizer(self.d_learning_rate, FLAGS.gd_gamma, name='d_optimizer',epsilon=1e-8)

            # Increment global step
            self.increment_global_step = tf.assign_add(self.global_step, 1)

            self.wikiart_batches = utils.get_wikiart_batches(FLAGS.data_dir, config.batch_size, FLAGS.image_size,
                                                             shuffle_buffer_size=config.shuffle_buffer_size)
            self.batches = self.wikiart_batches
            sample_images, _ = self.batches
            vis_images = tf.cast((sample_images + 1.) * 127.5, tf.uint8)
            tf.summary.image('input_image_grid',
                             tfgan.eval.image_grid(
                                 vis_images[:config.batch_size],
                                 grid_shape=utils.squarest_grid_size(config.batch_size),
                                 image_shape=(self.image_size, self.image_size)))

            images, sparse_labels = self.batches

            # sparse_labels = tf.squeeze(sparse_labels)
            print('han sparse_labels.shape', sparse_labels.shape)

            # Create indexing sequence
            id_seq = tf.range(0, config.batch_size, 1)
            label_indices = tf.stack([id_seq, sparse_labels], axis=1)

            gen_class_logits = tf.zeros((config.batch_size, FLAGS.n_classes))
            gen_class_ints = tf.multinomial(gen_class_logits, 1, output_dtype=tf.int32)
            gen_sparse_class = tf.squeeze(gen_class_ints)
            gen_sparse_indices = tf.stack([id_seq, gen_sparse_class], axis=1)
            print('han gen_sparse_class.shape', gen_sparse_class.shape)
            assert len(gen_class_ints.get_shape()) == 2
            gen_class_ints = tf.squeeze(gen_class_ints)
            assert len(gen_class_ints.get_shape()) == 1
            gen_class_vector = tf.one_hot(gen_class_ints, FLAGS.n_classes)
            assert len(gen_class_vector.get_shape()) == 2
            assert gen_class_vector.dtype == tf.float32

            # model operations
            if FLAGS.use_cin:
                self.gen_fcn = gen_module.conditional_instance_normalized_generator
                # generator generates images
                generator = self.gen_fcn(self.z_g_dist, gen_sparse_class, FLAGS.n_classes, FLAGS.gf_dim,
                                         FLAGS.n_stages)
            else:
                self.gen_fcn = gen_module.condition_embedded_generator
                # generator generates images
                generator = self.gen_fcn(self.z_g_dist, gen_sparse_class, FLAGS.z_g_dim, FLAGS.n_classes, FLAGS.gf_dim,
                                         FLAGS.n_stages)

            # define discriminator
            self.disc_fcn = disc_module.conditioned_discriminator

            assert generator.get_shape()[1] == FLAGS.image_size

            # discriminator classifies
            disc_on_data_logits, disc_on_data_mus, disc_on_data_sigmas = self.disc_fcn(images, label_indices,
                                                                                       FLAGS.n_classes,
                                                                                       FLAGS.df_dim, FLAGS.n_stages)

            # for test purposes: what happens if generator is apllied on 'common' discriminator version
            disc_on_gen_logits, disc_on_gen_mus, disc_on_gen_sigmas, disc_on_gen_mu_activations = self.disc_fcn(
                generator, gen_sparse_indices,
                FLAGS.n_classes,
                FLAGS.df_dim, FLAGS.n_stages, activate_mean=True)

            # image logging for generated data points
            vis_generator = tf.cast((generator + 1.) * 127.5, tf.uint8)
            tf.summary.image('generator_images', vis_generator)

            tf.summary.image('generator_grid',
                             tfgan.eval.image_grid(
                                 vis_generator[:config.batch_size],
                                 grid_shape=utils.squarest_grid_size(config.batch_size),
                                 image_shape=(self.image_size, self.image_size)))

            # compute losses
            # common discriminator loss
            loss_disc_on_real = _get_disc_on_real_loss(disc_on_data_logits)
            loss_disc_on_gen = _get_disc_on_gen_loss(disc_on_gen_logits)

            # bottleneck loss
            bottleneck_loss = _bottleneck_loss(tf.concat([disc_on_data_mus, disc_on_gen_mus], axis=0),
                                               tf.concat([disc_on_data_sigmas, disc_on_gen_sigmas], axis=0),
                                               self.information_constraint)

            # aggregate losses
            if FLAGS.use_gradient_penalty:
                gamma_gp = FLAGS.gp_gamma
                gradient_penalty = _gradient_penalty(disc_on_data_logits, images, gamma_gp)
                disc_loss = 0.5 * (
                            loss_disc_on_gen + loss_disc_on_real) + self.beta * bottleneck_loss + gradient_penalty

                # add summary for gradient penalty
                tf.summary.scalar('gradient_penalty', gradient_penalty)
            else:
                disc_loss = 0.5 * (loss_disc_on_gen + loss_disc_on_real) + self.beta * bottleneck_loss

            gen_loss = _gen_loss(disc_on_gen_mu_activations)

            # log logit losses
            disc_on_real_logit = tf.reduce_mean(disc_on_data_logits)
            disc_on_gen_logit = tf.reduce_mean(disc_on_gen_logits)

            # log discriminator mus and sigmas and activated mu's for the generator
            disc_mu_log = tf.reduce_mean(disc_on_data_mus)
            disc_sigma_log = tf.reduce_mean(disc_on_data_mus)
            gen_mu_log = tf.reduce_mean(disc_on_gen_mu_activations)

            # summaries
            tf.summary.scalar('discriminator_loss', disc_loss)
            tf.summary.scalar('loss_discriminator_on_real_data', loss_disc_on_real)
            tf.summary.scalar('loss_discriminator_on_fake_data', loss_disc_on_gen)
            tf.summary.scalar('bottleneck_loss', bottleneck_loss)
            tf.summary.scalar('generator_loss', gen_loss)
            tf.summary.scalar('discriminator_on_real_logits', disc_on_real_logit)
            tf.summary.scalar('discriminator_on_fake_logits', disc_on_gen_logit)
            tf.summary.scalar('d_learning_rate', self.d_learning_rate)
            tf.summary.scalar('g_learning_rate', self.g_learning_rate)
            tf.summary.scalar('disc_bottleneck_sigmas', disc_sigma_log)
            tf.summary.scalar('disc_bottleneck_mus', disc_mu_log)
            tf.summary.scalar('gen_mu_activations', gen_mu_log)

            # define variables for alternating gradient descent
            self._get_vars()

            # compute gradients
            d_grads = self.d_opt.compute_gradients(disc_loss, var_list=self.d_vars)
            g_grads = self.g_opt.compute_gradients(gen_loss, var_list=self.g_vars)

            train_summary = []

            # log gradients for generator
            for g, v in g_grads:
                if g is not None:
                    # print(format(v.name))
                    grad_hist_summary = tf.summary.histogram("{}/grad_histogram".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    train_summary.append(grad_hist_summary)
                    train_summary.append(sparsity_summary)
            tf.summary.merge(train_summary)

            # update ops should contain the averages for mean and variance of conditional instance normalization ( if enabled,
            # as well as update for the beta parameter of the information bottleneck
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print('Number of update_ops: ', len(update_ops))

            with tf.control_dependencies(update_ops):
                # apply gradients for discriminator before updating beta
                self.d_optim = self.d_opt.apply_gradients(d_grads)
                # make sure, the update operations have been executed before computing gradients (only important,
                # if conditional instance norm is used; else, this statement will have no effect)
                self.g_optim = self.g_opt.apply_gradients(g_grads)


            with tf.control_dependencies([self.d_optim]):
                # update beta
                update_beta = tf.assign(self.beta,tf.maximum(tf.constant(0, dtype=tf.float32),
                                       self.beta + self.beta_step_size * bottleneck_loss),name='update_beta')

            # log beta
            tf.summary.scalar('optimized_bottleneck_loss_weight', update_beta)

    def _get_vars(self):
        '''
        Define variables to adapt during gradient descent, both for discriminator and generator
        '''
        all_vars = tf.trainable_variables()
        # get discriminator trainable variables
        self.d_vars = [var for var in all_vars if var.name.startswith('discriminator')]
        # self.d_vars = tf.get_collection(all_vars,scope='generator')
        # get generator trainable variables
        self.g_vars = [var for var in all_vars if var.name.startswith('generator')]
        # self.d_vars = tf.get_collection(all_vars, scope='discriminator')
        # check validity
        for x in self.d_vars:
            assert x not in self.g_vars
        for x in self.g_vars:
            assert x not in self.d_vars
        for x in all_vars:
            assert x in self.g_vars or x in self.d_vars, x.name

        n_params = 0
        for var in self.d_vars:
            n_params += utils.prod(var.get_shape().as_list())

        print('Discriminator has {} parameters that are optimized during training'.format(n_params))

        n_params = 0
        for var in self.g_vars:
            n_params += utils.prod(var.get_shape().as_list())

        print('Generator has {} parameters that are optimized during training'.format(n_params))
