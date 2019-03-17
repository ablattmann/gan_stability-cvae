import os
import utils
import datetime

gpu = str(utils.choose_gpu())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import tensorflow as tf

from gan import VDBGAN
from eval_model import Evaluator
from absl import flags

tfgan = tf.contrib.gan
gfile = tf.gfile

train_dir = os.path.dirname(os.path.realpath(__file__))


flags.DEFINE_string('checkpoint_dir', os.path.join(train_dir, '../../data/wiki-art/vdbgan/checkpoints'),
                    'Directory, where the data to feed is located.')
flags.DEFINE_integer('batch_size', 32, 'The batch_size for the model.')
flags.DEFINE_integer('shuffle_buffer_size', 70000, 'Number of records to load '
                                                   'before shuffling and yielding for consumption. [100000]')
flags.DEFINE_integer('save_summaries_steps', 300, 'Number of seconds between '
                                                  'saving summary statistics. [1]')  # default 300
flags.DEFINE_integer('save_checkpoint_secs', 1200, 'Number of seconds between '
                                                   'saving checkpoints of model. [1200]')
flags.DEFINE_integer('d_steps_per_g_step', 1, 'Number of discriminator steps to take per generator step.')
flags.DEFINE_integer('d_step', 1, 'Number of discriminator steps to take per iteration.')
flags.DEFINE_integer('g_step', 1, 'Number of generator steps to take per iteration.')
flags.DEFINE_integer('n_steps', 3000000, 'The total number of train steps to take (reference is always generator).')
flags.DEFINE_bool('load_model', False, 'Whether to load from existing weights or train new from scratch.')



FLAGS = flags.FLAGS


def main(_):
    print('d_learning_rate', FLAGS.d_learning_rate)
    print('g_learning_rate', FLAGS.g_learning_rate)
    print('data_dir', FLAGS.data_dir)
    print(FLAGS.batch_size, FLAGS.gd_gamma)
    print('gf_df_dim', FLAGS.gf_dim, FLAGS.df_dim)
    print('Use conditional imstance normalization? ', FLAGS.use_cin)
    print('Starting the program..')
    gfile.MakeDirs(FLAGS.checkpoint_dir)


    if FLAGS.load_model:
        model_dir = 'vdbgan_wiki-art_32_128_2019-01-10-08-12-44'
        logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)
        print('reloading existing model and continuing training.')
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_dir = '%s_%s_%s_%s_%s' % (
            'vdbgan_wiki-art', FLAGS.batch_size, FLAGS.information_upper_bound, FLAGS.image_size, now)
        logdir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    print('checkpoint_dir: {}'.format(FLAGS.checkpoint_dir))
    print('model_dir: {}'.format(model_dir))

    gfile.MakeDirs(logdir)

    # for convenience
    with tf.Graph().as_default():
        # Start building up model

        # Set up device to use
        device = '/gpu:0'

        with tf.device(device):
            # Instantiate global_step.
            global_step = tf.train.create_global_step()

        # Create noise tensors
        zs = utils.make_z_normal(FLAGS.batch_size, FLAGS.z_g_dim, 'z_train')

        print('save_summaries_steps', FLAGS.save_summaries_steps)

        # instantiate model and build graph
        vdbgan = VDBGAN(zs, global_step, FLAGS, device=device)

        evaluator = Evaluator(vdbgan.gen_fcn,FLAGS,device,logdir)

        train_ops = tfgan.GANTrainOps(
            generator_train_op=vdbgan.g_optim,
            discriminator_train_op=vdbgan.d_optim,
            global_step_inc_op=vdbgan.increment_global_step)

        # Set allow_soft_placement to be True to avoid displacement of saver and some other ops on GPU
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        print("G step: ", FLAGS.g_step)
        print("D_step: ", FLAGS.d_step)

        train_steps = tfgan.GANTrainSteps(FLAGS.g_step, FLAGS.d_step)

        tfgan.gan_train(
            train_ops,
            get_hooks_fn=tfgan.get_sequential_train_hooks(
                train_steps=train_steps),
            hooks=
            [tf.train.StopAtStepHook(num_steps=FLAGS.n_steps),evaluator.compute_stats()],
            logdir=logdir,
            save_summaries_steps=FLAGS.save_summaries_steps,
            save_checkpoint_secs=FLAGS.save_checkpoint_secs,
            config=session_config)


if __name__ == '__main__':
    tf.app.run()
